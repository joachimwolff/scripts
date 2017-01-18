#! /usr/bin/env python

from bioblend import galaxy
import argparse

from manager import Manager
from users import Users
from groups import Groups
from jobs import Jobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="User statistics of a galaxy instance.")
    parser.add_argument("--instance", dest='instance', help="The galaxy instance url. E.g. http://galaxy.uni-freiburg.de or http://localhost:8080")
    parser.add_argument("--api-key", dest='api_key', help="The API-Key of a galaxy user to get access to the user statistics. in the most cases, admin rights are necessary.")
    
    parser.add_argument("--all", dest='all', help="Create all statistics")

    args = parser.parse_args()
    manager = Manager(args.instance, args.api_key)
    users = Users(manager.getGalaxyInstance())
    print users.get_number_of_users()
    print users.get_registered_users_ordered_per_year()
    groups = Groups(manager.getGalaxyInstance())
    print groups.get_group_by_name("Uniklinik")
    print groups.get_users_of_group("Uniklinik")
    # jobs = Jobs(manager)
    # print jobs.get_jobs()
