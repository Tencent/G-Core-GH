import pytest

from gpatch_v4.configs.profile_config import ProfileConfig


def test_use_nsys_default_false():
    assert ProfileConfig().use_nsys is False


def test_use_nsys_validates_profile_steps():
    with pytest.raises(ValueError, match="profile_start_step"):
        ProfileConfig(use_nsys=True, profile_start_step=3, profile_end_step=1)
