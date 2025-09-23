from .landmarkMap import Landmark, LandmarkMap

class LandmarkObservation(Landmark):
    def __init__(self, pose, signature, sigma):
        super().__init__(pose, signature)
        self.sigma = sigma
        self.credebility = 1

class ParticleMap(LandmarkMap):
    def __init__(self, landmarks: list[LandmarkObservation], roi: tuple[tuple[float, float], tuple[float, float]]):
        super().__init__(landmarks)
        self.roi = roi

    def getLandmarkBySignature(self, signature: int):
        for lm in self.landmarks:
            if lm.s == signature: return lm

    def addLandmark(self, landmark: LandmarkObservation):
        # check if signature is unique
        for lm in self.landmarks:
            if landmark.s == lm.s: raise ValueError(f"landmark {landmark} has same signature as existing {lm}")
        
        self.landmarks.append(landmark)
        self.colors = self.generateUniqueColors(len(self.landmarks))

    def removeLandmark(self, landmark: LandmarkObservation):
        self.landmarks.remove(landmark)
        self.colors.remove(self.colors[0])