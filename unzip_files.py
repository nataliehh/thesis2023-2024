import zipfile
zip_file = 'CNGT_isolated_signers' #'Signbank-NGT-videos.zip'
path = '/vol/tensusers5/nhollain/'
with zipfile.ZipFile(path + zip_file + '.zip', 'r') as zip_ref:
    zip_ref.extractall(path + zip_file)
