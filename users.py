class Users():
    
    def __init__(self, galaxy_manager):
        self._galaxy_manager = galaxy_manager
    
    def get_number_of_users(self):
        return len(self._galaxy_manager.users.get_users())
    def get_registered_users_ordered_per_year(self):
        pass
        # return self._galaxy_manager.users.show_user(user_id=)
        # self._galaxy_manager.users.
    def get_number_of_jobs_per_user(self, user_id):
        pass
    