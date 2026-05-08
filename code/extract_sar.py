import zipfile
import os

def extract_sar():
    output_dir = raw_dir

    if not os.path.exists(output_dir):
        return
    
    for file in os.listdir(output_dir):
        if file.endswith('.zip'):
            path = os.path.join(output_dir, file)
            safe_name = file.replace('.zip', '.SAFE')
            safe_path = os.path.join(output_dir, safe_name)

            if os.path.exists(safe_path):
                print(f'Skipping {file}')
                continue

            try:
                with zipfile.ZipFile(path, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)

                print(f'Extracted {file}')
                os.remove(path)

            except zipfile.BadZipFile:
                print(f'The file {file} is corrupted')

            except Exception as e:
                print(f'Failed to extract {file} with an exception {e}')

if __name__ == '__main__':
    raw_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'raw')