import os

def remove_file_from_subdirectories(file_name):
    """
    Find the file in the subdirectories of the root folder.
    Takes user input before deleting the file.
    """
    dir_=os.path.dirname(__file__)
    os.chdir(dir_)
    for root,dirs, files in os.walk("."):
        for file in files:
            if file==file_name:
                path=os.path.join(root,file)
                print(path)
                user_input=input("should the file be removed?:")
                if str(user_input)=='yes':
                    os.remove(path)

# remove_file_from_subdirectories('key.json')