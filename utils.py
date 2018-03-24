import sys

def = config_pahts(user, env_name):
    paths = ['',
         '/home/{0}/{1}/.env/bin'.format(user, env_name),
         '/usr/lib/python35.zip',
         '/usr/lib/python3.5',
         '/usr/lib/python3.5/plat-x86_64-linux-gnu',
         '/usr/lib/python3.5/lib-dynload',
         '/home/{0}/{1}/.env/lib/python3.5/site-packages'.format(user, env_name),
         '/home/{0}/{1}/.env/lib/python3.5/site-packages/IPython/extensions'.format(user, env_name),
         '/home/{0}/.ipython']

    for path in paths:
        sys.path.append(path)