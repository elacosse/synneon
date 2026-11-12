from synneon.neon.synthetic import synthetic_gaze_stream
from synneon.schemas.neons import GazeSample


async def _collect(n: int) -> list[GazeSample]:
    """Collect n samples from the synthetic stream."""
    samples = []
    async for sample in synthetic_gaze_stream():
        samples.append(sample)
        if len(samples) >= n:
            break
    return samples


class TestSyntheticGazeStream:
    async def test_yields_gaze_samples(self):
        samples = await _collect(5)
        assert all(isinstance(s, GazeSample) for s in samples)

    async def test_always_worn(self):
        samples = await _collect(20)
        assert all(s.is_worn for s in samples)

    async def test_x_clamped_to_unit_range(self):
        samples = await _collect(50)
        assert all(0.0 <= s.x <= 1.0 for s in samples)

    async def test_y_clamped_to_unit_range(self):
        samples = await _collect(50)
        assert all(0.0 <= s.y <= 1.0 for s in samples)

    async def test_timestamps_monotonically_increasing(self):
        samples = await _collect(10)
        for a, b in zip(samples, samples[1:], strict=False):
            assert b.timestamp >= a.timestamp

    async def test_yields_quickly(self):
        """Should yield at least one sample without hanging."""
        samples = await _collect(1)
        assert len(samples) == 1

    async def test_independent_instances_have_own_state(self):
        """Two independent calls produce separate generators."""
        gen1 = synthetic_gaze_stream()
        gen2 = synthetic_gaze_stream()
        s1 = await gen1.__anext__()
        s2 = await gen2.__anext__()
        # Both yield valid GazeSamples
        assert isinstance(s1, GazeSample)
        assert isinstance(s2, GazeSample)
        await gen1.aclose()
        await gen2.aclose()
