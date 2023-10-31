import shutil
zip_name = 'CNGT_landmarks_HD' #'NGT_landmarks'
output_filename = zip_name
dir_name = '/vol/tensusers5/nhollain/{}/'.format(zip_name)
shutil.make_archive(output_filename, 'zip', dir_name)
