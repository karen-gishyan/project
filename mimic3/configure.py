def configure():
    """ Configuration for python to correctly run django."""
    import os
    import os, sys 
    import django 
    sys.path.append(os.path.split(sys.path[0])[0])
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "project.settings")
    django.setup()
