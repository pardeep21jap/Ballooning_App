from balloon_app.data_model import Balloon, BalloonSource, ReviewStatus
from balloon_app.learning import apply_learned_feedback, learn_from_balloon, memory_stats


def _balloon(text="R5"):
    return Balloon(raw_text=text, char_type="radius", source=BalloonSource.AUTO.value)


def test_edited_correction_is_reused(tmp_path):
    path = tmp_path / "memory.json"
    reviewed = _balloon()
    reviewed.status = ReviewStatus.EDITED.value
    reviewed.nominal = 5.25
    learn_from_balloon(reviewed, path)

    proposal = _balloon()
    proposal.nominal = 5.0
    kept, corrected, suppressed = apply_learned_feedback([proposal], path)
    assert (corrected, suppressed) == (1, 0)
    assert kept[0].nominal == 5.25


def test_two_rejections_suppress_exact_callout(tmp_path):
    path = tmp_path / "memory.json"
    rejected = _balloon("NOT A DIMENSION 12.5")
    rejected.status = ReviewStatus.REJECTED.value
    learn_from_balloon(rejected, path)
    learn_from_balloon(rejected, path)

    kept, corrected, suppressed = apply_learned_feedback([_balloon(" not a dimension   12.5 ")], path)
    assert kept == []
    assert (corrected, suppressed) == (0, 1)
    assert memory_stats(path) == (0, 1)


def test_single_rejection_does_not_suppress(tmp_path):
    path = tmp_path / "memory.json"
    rejected = _balloon()
    rejected.status = ReviewStatus.REJECTED.value
    learn_from_balloon(rejected, path)
    kept, _, suppressed = apply_learned_feedback([_balloon()], path)
    assert len(kept) == 1
    assert suppressed == 0
