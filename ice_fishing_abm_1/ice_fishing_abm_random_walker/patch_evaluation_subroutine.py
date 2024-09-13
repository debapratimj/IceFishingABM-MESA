class PatchEvaluationSubroutine:
    def __init__(self, **kwargs):
        # TODO: implement parameters
        pass

    def stay_on_patch(self, time_on_patch, time_since_last_catch, threshold):
        if (time_on_patch > threshold) and (time_since_last_catch > threshold):
            return False
        return True


class GPPatchEvaluationSubroutine(PatchEvaluationSubroutine):
    def __init__(self):
        super().__init__()
        pass

    def stay_on_patch(self, time_on_patch, time_since_last_catch, threshold):
        return super().stay_on_patch(time_on_patch, time_since_last_catch, threshold)
