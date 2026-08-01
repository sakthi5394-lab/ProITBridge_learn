import os

def save_file(file, upload_folder):
    filename = file.filename
    path = os.path.join(upload_folder, filename)
    file.save(path)
    return filename, path