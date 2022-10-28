"""
This script downloads the SINS database to a local path.
It also allows to only download data for certain nodes.
Information about the database can be found here:
https://github.com/KULeuvenADVISE/SINS_database

Example usage:
download all nodes:
python3 download.py -db /destination/path/to/db

download living room nodes:
python3 download.py -db /destination/path/to/db -n 1,2,3,4,6,7,8
"""
import argparse
import os
import sys
import shutil
from pathlib import Path

from paderbox.io.download import (
    download_file_list,
    download_file,
    extract_file
)
from tqdm import tqdm

RECORDS = {
    '1': '2546677',
    '2': '2547307',
    '3': '2547309',
    '4': '2555084',
    '6': '2547313',
    '7': '2547315',
    '8': '2547319',
    '9': '2555080',
    '10': '2555137',
    '11': '2558362',
    '12': '2555141',
    '13': '2555143'
}


def download_node(node_id, database_path):
    """
    Download data from single node to database_path

    Args:
        node_id:
        database_path:

    Returns:

    """

    files = [
        f'https://zenodo.org/record/{RECORDS[node_id]}/files/{file}'
        for file in [
            'Node{}_audio_{:02}.zip'.format(node_id, j) if int(node_id) < 10
            else 'Node{}_audio_{}.zip'.format(node_id, j)
            for j in range(1, 10 + (int(node_id) < 9))
        ] + ['license.pdf'] + (['readme.txt'] if node_id not in ['1', '8'] else [])
    ]
    download_file_list(files, database_path, exist_ok=True)


def download(nodes, database_path):
    """
    Download dataset over the internet to the local path

    Args:
        nodes: list of nodes to download
        database_path: local path to store database at.

    Returns:

    """
    database_path = Path(database_path)
    os.makedirs(database_path, exist_ok=True)

    # Download git repository with annotations and matlab code
    extract_file(download_file(
        'https://github.com/KULeuvenADVISE/SINS_database/archive/master.zip',
        database_path / 'SINS_database-master.zip'
    ))

    # Download node data from zenodo repositories
    item_progress = tqdm(
        nodes, desc="{0: <25s}".format('Download nodes'),
        file=sys.stdout, leave=False, ascii=True)

    for node_id in item_progress:
        download_node(node_id, database_path)
    shutil.move(
        str(database_path / 'Original' / 'audio'),
        str(database_path / 'audio')
    )
    os.rmdir(str(database_path / 'Original'))
    for file in (database_path / 'SINS_database-master').iterdir():
        shutil.move(str(file), str(database_path))
    os.rmdir(str(database_path / 'SINS_database-master'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download SINS.')
    parser.add_argument(
        '--nodes', '-n', default='1,2,3,4,6,7,8,9,10,11,12,13',
        help='String of desired nodes separated by ","'
    )
    parser.add_argument(
        '--database_path', '-db',
        default=os.environ['SINS_DB_DIR'] if "SINS_DB_DIR" in os.environ
        else './SINS',
        help='Local path to store SINS database.'
    )
    args = parser.parse_args()
    download(args.nodes.split(',') if args.nodes else [], args.database_path)
