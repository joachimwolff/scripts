from bioblend import galaxy

class Manager():
    def __init__(self, instance, api_key):
        self._gi = galaxy.GalaxyInstance(url=instance, key=api_key)
    def getGalaxyInstance(self):
        return self._gi