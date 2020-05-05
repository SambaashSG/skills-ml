import boto
import boto3
import json
from moto import mock_s3, mock_s3_deprecated
import unittest
from unittest.mock import patch
import importlib
import inspect
import functools

from tests import utils

from skills_ml.job_postings.computed_properties.computers import (
    PostingIdPresent,
    TitleCleanPhaseOne,
    TitleCleanPhaseTwo,
    Geography,
    SOCClassifyProperty,
    SkillCounts,
    YearlyPay
)

from skills_ml.algorithms.skill_extractors import ExactMatchSkillExtractor
from skills_ml.job_postings.geography_queriers.base import JobGeographyQuerier

from skills_ml.storage import S3Store


class ComputedPropertyTestCase(unittest.TestCase):
    datestring = '2016-01-01'

    def test_aggregator_compatibility(self):
        """Test whether or not the computer's compatible_aggregate_function_paths
            are in fact compatible with pandas.DataFrame.agg
        """
        computed_property = getattr(self, 'computed_property', None)
        if not computed_property:
            if self.__class__.__name__ == 'ComputedPropertyTestCase':
                return
            else:
                raise ValueError('All subclasses of ComputedPropertyTestCase ' +
                                 'should create self.computed_property in self.setUp')

        df = self.computed_property.df_for_key(self.datestring)

        def pandas_ready_functions(paths):
            """Generate aggregate functions from the configured aggregate function paths

            Suitable for testing whether or not the functions work with pandas

            Yields: callables
            """
            if paths:
                for path in paths.keys():
                    module_name, func_name = path.rsplit(".", 1)
                    module = importlib.import_module(module_name)
                    func = getattr(module, func_name)
                    # skills_ml.algorithms.aggregators.pandas functions are wrapped
                    if hasattr(func, 'function'):
                        base_func = func.function
                    else:
                        base_func = func

                    # assuming here that the first arg will be a configurable number,
                    # e.g. top 'n'
                    if len(inspect.getfullargspec(base_func).args) == 2:
                        yield functools.partial(func, 2)
                    else:
                        yield func

        for column in self.computed_property.property_columns:
            for func in pandas_ready_functions(column.compatible_aggregate_function_paths):
                df.agg(func)


@mock_s3
class PostingIdPresentTest(ComputedPropertyTestCase):
    def setUp(self):
        self.client = boto3.resource('s3')
        self.client.create_bucket(Bucket='test-bucket')
        self.storage = S3Store('s3://test-bucket/computed_properties')
        self.computed_property = PostingIdPresent(self.storage)
        self.job_postings = [utils.job_posting_factory(datePosted=self.datestring)]
        self.computed_property.compute_on_collection(self.job_postings)

    def test_compute_func(self):
        cache = self.computed_property.cache_for_key(self.datestring)
        job_posting_id = self.job_postings[0]['id']
        assert cache[str(job_posting_id)] == 1

    def test_sum(self):
        pass


@mock_s3
class TitleCleanPhaseOneTest(ComputedPropertyTestCase):
    def setUp(self):
        self.client = boto3.resource('s3')
        self.client.create_bucket(Bucket='test-bucket')
        self.storage = S3Store('s3://test-bucket/computed_properties')
        self.computed_property = TitleCleanPhaseOne(self.storage)
        self.job_postings = [utils.job_posting_factory(datePosted=self.datestring, title='Software Engineer - Tulsa')]
        self.computed_property.compute_on_collection(self.job_postings)

    def test_compute_func(self):
        cache = self.computed_property.cache_for_key(self.datestring)
        job_posting_id = self.job_postings[0]['id']
        assert cache[str(job_posting_id)] == 'software engineer tulsa'


@mock_s3
class TitleCleanPhaseTwoTest(ComputedPropertyTestCase):
    def setUp(self):
        self.client = boto3.resource('s3')
        self.client.create_bucket(Bucket='test-bucket')
        self.storage = S3Store('s3://test-bucket/computed_properties')
        self.computed_property = TitleCleanPhaseTwo(self.storage)
        self.job_postings = [utils.job_posting_factory(datePosted=self.datestring, title='Software Engineer Tulsa')]
        with patch('skills_ml.algorithms.jobtitle_cleaner.clean.negative_positive_dict', return_value={'places': ['tulsa'], 'states': [], 'onetjobs': ['software engineer']}):
            self.computed_property.compute_on_collection(self.job_postings)

    def test_compute_func(self):
        cache = self.computed_property.cache_for_key(self.datestring)
        job_posting_id = self.job_postings[0]['id']
        assert cache[str(job_posting_id)] == 'software engineer'



@mock_s3
class GeographyTest(ComputedPropertyTestCase):
    def setUp(self):
        client = boto3.resource('s3')
        bucket = client.create_bucket(Bucket='test-bucket')
        storage = S3Store('s3://test-bucket/computed_properties')
        cache_storage = S3Store('s3://test-bucket')
        class SampleJobGeoQuerier(JobGeographyQuerier):
            name = 'blah'
            output_columns = (
                ('city', 'the city'),
            )
            def _query(self, job_posting):
                return ['Fargo']
        self.computed_property = Geography(
            geo_querier=SampleJobGeoQuerier(),
            storage=storage,
        )
        self.job_postings = [utils.job_posting_factory(datePosted=self.datestring)]
        self.computed_property.compute_on_collection(self.job_postings)

    def test_compute_func(self):
        cache = self.computed_property.cache_for_key(self.datestring)
        job_posting_id = self.job_postings[0]['id']
        assert cache[job_posting_id] == ['Fargo']


@mock_s3
class SocClassifyWithFakeClassifierTest(ComputedPropertyTestCase):
    def setUp(self):
        client = boto3.resource('s3')
        bucket = client.create_bucket(Bucket='test-bucket')
        storage = S3Store('s3://test-bucket/computed_properties')
        description = 'This is my description'
        class MockClassifier(object):
            def predict_soc(self, document):
                assert document.strip() == description.lower()
                return '11-1234.00'

            @property
            def name(self):
                return "MockClassifier"

            @property
            def description(self):
                return "fake algorithm"

        self.computed_property = SOCClassifyProperty(
            storage=storage,
            classifier_obj=MockClassifier(),
        )
        self.job_postings = [utils.job_posting_factory(datePosted=self.datestring, description=description, skills='', qualifications='', experienceRequirements='')]
        self.computed_property.compute_on_collection(self.job_postings)

    def test_compute_func(self):
        cache = self.computed_property.cache_for_key(self.datestring)
        job_posting_id = self.job_postings[0]['id']
        assert cache[job_posting_id] == '11-1234.00'

    def test_name_description(self):
        assert self.computed_property.property_name == "soc_mock_classifier"
        assert self.computed_property.property_description == "SOC code classifier using fake algorithm"

@mock_s3
@mock_s3_deprecated
class SkillExtractTest(ComputedPropertyTestCase):
    def setUp(self):
        s3_conn = boto.connect_s3()
        client = boto3.resource('s3')
        bucket = client.create_bucket(Bucket='test-bucket')
        storage = S3Store('s3://test-bucket/computed_properties')
        skill_extractor = ExactMatchSkillExtractor(utils.sample_framework())
        self.computed_property = SkillCounts(
            skill_extractor=skill_extractor,
            storage=storage,
        )
        self.job_postings = [utils.job_posting_factory(
            datePosted=self.datestring,
            description='reading comprehension'
        )]
        self.computed_property.compute_on_collection(self.job_postings)

    def test_compute_func(self):
        cache = self.computed_property.cache_for_key(self.datestring)
        job_posting_id = self.job_postings[0]['id']
        assert cache[job_posting_id] == {'skill_counts_sample_framework_exact_match': ['reading comprehension']}


@mock_s3
class YearlyPayTest(ComputedPropertyTestCase):
    def setUp(self):
        self.client = boto3.resource('s3')
        self.client.create_bucket(Bucket='test-bucket')
        self.storage = S3Store('s3://test-bucket/computed_properties')
        self.computed_property = YearlyPay(self.storage)
        self.job_postings = [utils.job_posting_factory(
            id=5,
            datePosted=self.datestring,
            baseSalary={'salaryFrequency': 'yearly', 'minValue': 5, 'maxValue': ''}
        ), utils.job_posting_factory(
            id=6,
            datePosted=self.datestring,
            baseSalary={'salaryFrequency': 'yearly', 'minValue': '6.25', 'maxValue': '9.25'}
        )]
        self.computed_property.compute_on_collection(self.job_postings)

    def test_compute_func(self):
        cache = self.computed_property.cache_for_key(self.datestring)
        assert cache[str(self.job_postings[0]['id'])] == 5
        assert cache[str(self.job_postings[1]['id'])] == 7.75
