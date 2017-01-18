class Groups():

    def __init__(self, galaxy_manager):
        self._galaxy_manager = galaxy_manager

    def get_groups(self):
        return self._galaxy_manager.groups.get_groups()
    def get_group_by_name(self, pName):
        return_list = []
        groups_dict = self._galaxy_manager.groups.get_groups()
        for group in groups_dict:
            if pName in group['name']:
                return_list.append(group)
        return return_list
    def get_users_of_group(self, pGroup_name):
        groups_dict = self._galaxy_manager.groups.get_groups()
        for group in groups_dict:
            if pGroup_name in group['name']:
                print group['name']
                return self._galaxy_manager.groups.get_group_users(group['id'])