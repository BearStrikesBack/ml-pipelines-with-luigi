import logging

import luigi
import os
from pathlib import Path

from util import DockerTask

VERSION = os.getenv('PIPELINE_VERSION', '0.1')


class Debug(DockerTask):
    """Use this task with appropriate image to debug things."""

    @property
    def image(self):
        return f'code-challenge/download-data:{VERSION}'

    @property
    def command(self):
        return [
            'sleep', '3600'
        ]


class DownloadData(DockerTask):
    """Initial pipeline task downloads dataset."""

    fname = luigi.Parameter(default='wine_dataset')
    out_dir = luigi.Parameter(default='/usr/share/data/raw/')
    url = luigi.Parameter(
        default='https://github.com/datarevenue-berlin/code-challenge-2019/'
                'releases/download/0.1.0/dataset_sampled.csv'
    )

    @property
    def image(self):
        return f'code-challenge/download-data:{VERSION}'

    @property
    def command(self):
        return [
            'python', 'download_data.py',
            '--name', self.fname,
            '--url', self.url,
            '--out-dir', self.out_dir
        ]

    def output(self):
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        return luigi.LocalTarget(
            path=str(out_dir/f'{self.fname}.csv')
        )


class ProcessData(DockerTask):
    """Pipeline task to clean and make train/test split."""

    in_csv = luigi.Parameter(default='/usr/share/data/raw/wine_dataset.csv')
    out_dir = luigi.Parameter(default='/usr/share/data/processed/')
    test_size = luigi.Parameter(default='0.3')

    @property
    def image(self):
        return f'code-challenge/process-data:{VERSION}'

    def requires(self):
        return DownloadData()

    @property
    def command(self):
        # TODO: implement correct command
        # Try to get the input path from self.requires() ;)
        return [
            'python', 'process_data.py',
            '--in-csv', self.in_csv,
            '--out-dir', self.out_dir,
            '--test-size', self.test_size
        ]

    def output(self):
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / '.SUCCESS')
        )

class TrainModels(DockerTask):
    """Pipeline task to train two models"""

    in_csv = luigi.Parameter(default='/usr/share/data/processed/train.csv')
    out_dir = luigi.Parameter(default='/usr/share/data/models/')

    @property
    def image(self):
        return f'code-challenge/train-models:{VERSION}'

    def requires(self):
        return ProcessData()
    
    @property
    def command(self):
        return [
            'python', 'train_models.py',
            '--in-csv', self.in_csv,
            '--out-dir', self.out_dir
        ]

    def output(self):
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / '.SUCCESS')
        )

class EvaluateModels(DockerTask):
    """Pipeline task to evaluate two models"""

    in_csv = luigi.Parameter(default='/usr/share/data/processed/test.csv')
    in_dir = luigi.Parameter(default='/usr/share/data/models/')
    out_dir = luigi.Parameter(default='/usr/share/data/results/')

    @property
    def image(self):
        return f'code-challenge/evaluate-models:{VERSION}'

    def requires(self):
        return TrainModels()
    
    @property
    def command(self):
        return [
            'python', 'evaluate_models.py',
            '--in-csv', self.in_csv,
            '--in-dir', self.in_dir,
            '--out-dir', self.out_dir
        ]

    def output(self):
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / '.SUCCESS')
        )

class MakeReport(DockerTask):
    """Pipeline task to make report"""
    
    out_dir = luigi.Parameter(default='/usr/share/data/report/')

    @property
    def image(self):
        return f'code-challenge/make-report:{VERSION}'

    def requires(self):
        return EvaluateModels()
    
    @property
    def command(self):
        return [
            'pweave', 'make_report.py',
            '--output', str(Path(self.out_dir) / 'report.html')
        ]

    def output(self):
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / 'report.html')
        )
