"""Independent CPU models for exposure-aware news retrieval and reranking.

News features are fixed, training-fitted TF-IDF/SVD/category vectors. They are
not learned item embeddings. Row zero is reserved for padding. No model uses
candidate-set normalization, so scores are independent of batch composition.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _validate_news(features, feature_dim):
    if features.ndim < 2 or features.shape[-1] != feature_dim:
        raise ValueError(f"Expected news features [..., {feature_dim}] with at least two dimensions")
    if not features.is_floating_point():
        raise ValueError("News features must be floating point")


def _history_mask(history_features, history_mask, feature_dim):
    if history_features.ndim != 3 or history_features.shape[-1] != feature_dim:
        raise ValueError(f"Expected history [batch, history, {feature_dim}]")
    if not history_features.is_floating_point():
        raise ValueError("History features must be floating point")
    non_padding = history_features.ne(0).any(dim=-1)
    if history_mask is None:
        return non_padding
    if history_mask.dtype != torch.bool or history_mask.shape != history_features.shape[:2]:
        raise ValueError("history_mask must be boolean [batch, history]")
    return history_mask & non_padding


def _masked_softmax(scores, mask):
    """Softmax on the final axis; an entirely masked row is exactly zero."""
    mask = torch.broadcast_to(mask, scores.shape)
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    probabilities = torch.softmax(scores, dim=-1) * mask.to(scores.dtype)
    return probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(scores.dtype).tiny
    )


class TwoTower(nn.Module):
    """News/user towers with masked mean or additive attention history pooling.

    Args:
        feature_dim: Dimension of the fixed news input vectors.
        embedding_dim: Dimension of normalized retrieval vectors.
        hidden: Hidden size of each MLP and optional attention scorer.
        user_pooling: ``mean`` or ``attention``. Mean variants have exactly the
            same model regardless of their externally implemented sampler.

    ``encode_news`` returns zero for PAD and unit vectors otherwise, except for
    a mathematically zero projection. ``encode_user`` returns zero for empty
    history; this explicitly leaves cold-history fallback to the serving layer.
    Learned user attention pools fixed features before the user MLP, avoiding
    repeated news-MLP evaluation over every historical item in every batch.
    """

    def __init__(self, feature_dim, embedding_dim=64, hidden=128, user_pooling="mean"):
        super().__init__()
        if feature_dim <= 0 or embedding_dim <= 0 or hidden <= 0:
            raise ValueError("Model dimensions must be positive")
        if user_pooling not in {"mean", "attention"}:
            raise ValueError("user_pooling must be 'mean' or 'attention'")
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.hidden = hidden
        self.user_pooling = user_pooling
        self.news_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.ReLU(), nn.Linear(hidden, embedding_dim)
        )
        self.user_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.ReLU(), nn.Linear(hidden, embedding_dim)
        )
        self.history_attention = (
            nn.Sequential(nn.Linear(feature_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1, bias=False))
            if user_pooling == "attention" else None
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def encode_news(self, news_features):
        _validate_news(news_features, self.feature_dim)
        active = news_features.ne(0).any(dim=-1, keepdim=True)
        vectors = F.normalize(self.news_encoder(news_features), dim=-1)
        return vectors * active.to(vectors.dtype)

    def encode_user(self, history_features, history_mask=None):
        mask = _history_mask(history_features, history_mask, self.feature_dim)
        clean_history = history_features.masked_fill(~mask.unsqueeze(-1), 0)
        if history_features.shape[1] == 0:
            pooled = history_features.new_zeros((history_features.shape[0], self.feature_dim))
        elif self.user_pooling == "mean":
            pooled = clean_history.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        else:
            scores = self.history_attention(clean_history).squeeze(-1)
            weights = _masked_softmax(scores, mask)
            pooled = torch.bmm(weights.unsqueeze(1), clean_history).squeeze(1)
        vectors = F.normalize(self.user_encoder(pooled), dim=-1)
        return vectors * mask.any(dim=1, keepdim=True).to(vectors.dtype)

    def forward(self, history_features, candidate_features, history_mask=None):
        user = self.encode_user(history_features, history_mask)
        news = self.encode_news(candidate_features)
        if news.shape[0] != user.shape[0] or news.ndim not in {2, 3}:
            raise ValueError("Candidates must have shape [batch, D] or [batch, candidates, D]")
        scores = (news * user).sum(dim=-1) if news.ndim == 2 else torch.einsum("bd,bcd->bc", user, news)
        return scores * self.logit_scale.clamp(min=0, max=math.log(100.0)).exp()


class AttentionRanker(nn.Module):
    """Candidate-conditioned history attention, trained on exposed candidates.

    ``candidate_prior_features`` is optional [B,C,2], provided by the caller
    from training-only statistics. Values are already transformed; this class
    does not fit, update or logarithmically transform them. Feature ordering is
    part of the caller's checkpoint manifest. Missing priors are exactly zero.

    The shared projection is linear, so history/candidate projection can be
    cached at inference when its model revision is fixed. During training these
    projected representations are recomputed to preserve parameter gradients.
    The ranker has no claim to be a new attention algorithm or full DIN/NRMS.
    """

    def __init__(self, feature_dim, hidden=64, embedding_dim=64):
        super().__init__()
        if feature_dim <= 0 or hidden <= 0 or embedding_dim <= 0:
            raise ValueError("Model dimensions must be positive")
        self.feature_dim = feature_dim
        self.hidden = hidden
        self.embedding_dim = embedding_dim
        self.projection = nn.Linear(feature_dim, embedding_dim, bias=False)
        self.attention_log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        # Four E-dimensional vectors plus dot/mean/max similarity, two priors,
        # and an indicator that distinguishes missing history from low similarity.
        self.scorer = nn.Sequential(
            nn.Linear(4 * embedding_dim + 6, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def score_candidates(self, history_features, history_mask, candidate_features,
                         candidate_prior_features=None):
        mask = _history_mask(history_features, history_mask, self.feature_dim)
        _validate_news(candidate_features, self.feature_dim)
        if candidate_features.ndim != 3 or candidate_features.shape[0] != history_features.shape[0]:
            raise ValueError("candidate_features must be [batch, candidates, feature_dim]")
        batch_size, candidate_count = candidate_features.shape[:2]
        if candidate_prior_features is None:
            priors = candidate_features.new_zeros((batch_size, candidate_count, 2))
        else:
            if candidate_prior_features.shape != (batch_size, candidate_count, 2):
                raise ValueError("candidate_prior_features must be [batch, candidates, 2]")
            priors = candidate_prior_features.to(device=candidate_features.device, dtype=candidate_features.dtype)
        clean_history = history_features.masked_fill(~mask.unsqueeze(-1), 0)
        candidate = F.normalize(self.projection(candidate_features), dim=-1)
        history = F.normalize(self.projection(clean_history), dim=-1)
        has_history = mask.any(dim=-1)
        if history_features.shape[1] == 0:
            attended = candidate.new_zeros(candidate.shape)
            mean_similarity = candidate.new_zeros((batch_size, candidate_count))
            max_similarity = mean_similarity.clone()
        else:
            similarities = torch.bmm(candidate, history.transpose(1, 2))
            expanded_mask = mask.unsqueeze(1)
            scale = self.attention_log_scale.clamp(min=0, max=math.log(100.0)).exp()
            weights = _masked_softmax(similarities * scale, expanded_mask)
            attended = torch.bmm(weights, history)
            mean_similarity = (similarities * expanded_mask).sum(dim=-1) / mask.sum(dim=-1, keepdim=True).clamp_min(1)
            max_similarity = similarities.masked_fill(~expanded_mask, torch.finfo(similarities.dtype).min).max(dim=-1).values
            max_similarity = torch.where(has_history.unsqueeze(1), max_similarity, torch.zeros_like(max_similarity))
        dot_similarity = (candidate * attended).sum(dim=-1)
        history_indicator = has_history[:, None].expand(batch_size, candidate_count).to(candidate.dtype)
        scalar_features = torch.stack((dot_similarity, mean_similarity, max_similarity, history_indicator), dim=-1)
        inputs = torch.cat((candidate, attended, candidate * attended, (candidate - attended).abs(),
                            scalar_features, priors), dim=-1)
        return self.scorer(inputs).squeeze(-1)

    def forward(self, history_features, candidate_features, history_mask=None,
                candidate_prior_features=None):
        """Convenience interface also accepting one candidate per user [B,D]."""
        single = candidate_features.ndim == 2
        if single:
            candidate_features = candidate_features.unsqueeze(1)
            if candidate_prior_features is not None:
                candidate_prior_features = candidate_prior_features.unsqueeze(1)
        scores = self.score_candidates(history_features, history_mask, candidate_features,
                                       candidate_prior_features)
        return scores.squeeze(1) if single else scores
