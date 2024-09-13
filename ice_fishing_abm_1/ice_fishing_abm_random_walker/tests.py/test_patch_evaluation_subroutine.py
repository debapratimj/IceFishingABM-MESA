from ice_fishing_abm_1.ice_fishing_abm_random_walker.patch_evaluation_subroutine import PatchEvaluationSubroutine


def test_patch_evaluation_subroutine_default():
    pes = PatchEvaluationSubroutine()

    assert pes.stay_on_patch(1, 10, 5) is True
    assert pes.stay_on_patch(5, 10, 5) is True
    assert pes.stay_on_patch(6, 10, 5) is False


