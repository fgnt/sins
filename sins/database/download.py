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
import logging
import os
import socket
import sys
import tarfile
import zipfile
from os import path
from urllib.request import urlretrieve

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


def download_file(remote_file, local_file):
    """
    Download single file to local_dir

    Args:
        remote_file:
        local_file:

    Returns:

    """
    local_dir = path.dirname(local_file)
    if not path.isfile(local_file):
        def progress_hook(t):
            """
            Wraps tqdm instance. Don't forget to close() or __exit__()
            the tqdm instance once you're done with it (easiest using
            `with` syntax).
            """

            last_b = [0]

            def inner(b=1, bsize=1, tsize=None):
                """
                b  : int, optional
                    Number of blocks just transferred [default: 1].
                bsize  : int, optional
                    Size of each block (in tqdm units) [default: 1].
                tsize  : int, optional
                    Total size (in tqdm units). If [default: None]
                    remains unchanged.
                """
                if tsize is not None:
                    t.total = tsize
                t.update((b - last_b[0]) * bsize)
                last_b[0] = b

            return inner

        tmp_file = path.join(local_dir, 'tmp_file')
        with tqdm(
                desc="{0: >25s}".format(
                    path.splitext(remote_file.split('/')[-1])[0]),
                file=sys.stdout,
                unit='B',
                unit_scale=True,
                miniters=1,
                leave=False,
                ascii=True
        ) as t:

            urlretrieve(
                remote_file,
                filename=tmp_file,
                reporthook=progress_hook(t),
                data=None
            )
        os.rename(tmp_file, local_file)
    return local_file


def extract_file(local_file):
    """
    extract files if local_file zip or tar.gz else do nothing

    Args:
        local_file:

    Returns:

    """

    local_dir = path.dirname(local_file)
    if path.isfile(local_file):

        if local_file.endswith('.zip'):

            with zipfile.ZipFile(local_file, "r") as z:
                # Trick to omit first level folder
                parts = []
                for name in z.namelist():
                    if not name.endswith('/'):
                        parts.append(name.split('/')[:-1])
                prefix = path.commonprefix(parts) or ''

                if prefix:
                    if len(prefix) > 1:
                        prefix_ = list()
                        prefix_.append(prefix[0])
                        prefix = prefix_

                    prefix = '/'.join(prefix) + '/'
                offset = len(prefix)

                # Start extraction
                members = z.infolist()
                for i, member in enumerate(members):
                    if len(member.filename) > offset:
                        member.filename = member.filename[offset:]
                        if not path.isfile(path.join(
                                local_dir, member.filename)):
                            try:
                                z.extract(member=member, path=local_dir)
                            except KeyboardInterrupt:
                                # Delete latest file, since most likely it
                                # was not extracted fully
                                os.remove(
                                    path.join(local_dir, member.filename)
                                )

                                # Quit
                                sys.exit()
            os.remove(local_file)

        elif local_file.endswith('.tar.gz'):
            tar = tarfile.open(local_file, "r:gz")
            for i, tar_info in enumerate(tar):
                if not path.isfile(
                        path.join(local_dir, tar_info.name)):
                    tar.extract(tar_info, local_dir)
                tar.members = []
            tar.close()
            os.remove(local_file)


def download_node(node_id, database_path, logger=None):
    """
    Download data from single node to database_path

    Args:
        node_id:
        database_path:
        logger:

    Returns:

    """

    files = [
        f'https://zenodo.org/record/{RECORDS[node_id]}/files/{file}'
        for file in [
            'Node{}_audio_{:02}.zip'.format(node_id, j) if int(node_id) < 10
            else 'Node{}_audio_{}.zip'.format(node_id, j)
            for j in range(1, 10 + (int(node_id) < 9))
        ] + ['license.pdf'] + (['readme.txt'] if node_id != '1' else [])
    ]
    os.makedirs(database_path, exist_ok=True)

    # Set socket timeout
    socket.setdefaulttimeout(120)

    item_progress = tqdm(
        files, desc="{0: <25s}".format('Download files'),
        file=sys.stdout, leave=False, ascii=True)

    local_files = list()
    for remote_file in item_progress:
        local_files.append(download_file(
            remote_file, path.join(database_path, path.basename(remote_file))
        ))

    item_progress = tqdm(local_files,
                         desc="{0: <25s}".format('Extract files'),
                         file=sys.stdout,
                         leave=False,
                         ascii=True)

    if logger is not None:
        logger.info('Starting Extraction')
    for _id, local_file in enumerate(item_progress):
        if local_file and path.isfile(local_file):
            if logger is not None:
                logger.info(
                    '  {title:<15s} [{item_id:d}/{total:d}] {package:<30s}'
                    .format(
                        title='Extract files ',
                        item_id=_id,
                        total=len(item_progress),
                        package=local_file))
            extract_file(local_file)


def download(nodes, database_path):
    """
    Download dataset over the internet to the local path

    Args:
        nodes: list of nodes to download
        database_path: local path to store database at.

    Returns:

    """
    logger = logging.getLogger('download')
    os.makedirs(database_path, exist_ok=True)

    # Set socket timeout
    socket.setdefaulttimeout(120)

    # Download git repository with annotations and matlab code
    extract_file(download_file(
        'https://github.com/KULeuvenADVISE/SINS_database/archive/master.zip',
        path.join(database_path, 'SINS_database-master.zip')
    ))

    # Download node data from zenodo repositories
    item_progress = tqdm(
        nodes, desc="{0: <25s}".format('Download nodes'),
        file=sys.stdout, leave=False, ascii=True)

    for node_id in item_progress:
        download_node(node_id, database_path, logger)


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
