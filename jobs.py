class Jobs():
    
    def __init__(self, galaxy_manager):
        self._galaxy_manager = galaxy_manager

    def get_jobs(self):
        return self._galaxy_manager.jobs.get_jobs()