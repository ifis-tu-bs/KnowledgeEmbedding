# -*- coding: utf-8 -*-
import ctypes

from pyke.library import Library
from pyke.parser import NTriplesParser


class Dataset(object):
    """
    Manages a collection of relational data
    encoded as a set of triples, each describing a statement (or fact)
    over two objects called 'entities', one 'head' and one 'tail',
    being related in a manner that is symbolized by a relation 'label'.

    The application encodes both entities and relations as integral values,
    describing an index in an ordered table.
    """

    def __init__(self, filename: str, temp_dir: str = ".pyke", generate_valid_test: bool = False,
                 fail_silently: bool = True):
        """
        Creates a new dataset from a N-triples file.

        .. note:

           The N-triples file is parsed into the original OpenKE benchmark file structure containing a file for the
           entities (entity2id.txt), for the relations (relation2id.txt) and the training file (train2id.txt). These
           files are stored by default in the `.pyke` directory in a subdirectory named after the MD5-sum of the
           input file. The MD5-sum is used to prevent the tool from recreating the benchmark files for.
           If you change the N-triples file the MD5-sum changes and so the entities and relations get a new id.

        .. note:

           At the moment, no two datasets can be open at the same time!

        :param filename: Pathname to the N-triples file for training
        :param temp_dir: Directory for storing the benchmark files. Application needs write access
        :param fail_silently: If true, ignore errors in file and skip lines
        """
        self.__library = Library.get_library(temp_dir)

        parser = NTriplesParser(filename, temp_dir, generate_valid_test, fail_silently)
        parser.parse()

        self.benchmark_dir = parser.output_dir if parser.output_dir[:-1] == "/" else parser.output_dir + "/"
        self.__library.setInPath(ctypes.create_string_buffer(self.benchmark_dir.encode(), len(self.benchmark_dir) * 2))
        self.__library.importTrainFiles()
        if generate_valid_test:
            self.__library.importTestFiles()
            self.__library.importTypeFiles()

        self.generate_valid_test = generate_valid_test
        self.size = parser.train_count
        self.ent_count = parser.ent_count
        self.rel_count = parser.rel_count
        self.shape = self.ent_count, self.rel_count
        self.train_set = self.read_benchmark(parser.train_file)
        self.test_set = self.read_benchmark(parser.test_file) if generate_valid_test else []
        self.valid_set = self.read_benchmark(parser.valid_file) if generate_valid_test else []
        self.entity2id = {}
        self.id2entity = {}
        self.relation2id = {}
        self.id2relation = {}
        with open(parser.entity_file) as f:
            f.readline()
            for line in f:
                try:
                    ent, ent_id = line.rsplit(maxsplit=1)
                    ent_id = int(ent_id)
                    self.entity2id[ent] = ent_id
                    self.id2entity[ent_id] = ent
                except ValueError:
                    continue
        with open(parser.relation_file) as f:
            f.readline()
            for line in f:
                try:
                    rel, rel_id = line.rsplit(maxsplit=1)
                    rel_id = int(rel_id)
                    self.relation2id[rel] = rel_id
                    self.id2relation[rel_id] = rel
                except ValueError:
                    continue

    def __len__(self):
        """Returns the size of the dataset."""
        return self.size

    def get_entity_id(self, entity):
        return self.entity2id[entity]

    def get_entity(self, eid):
        return self.id2entity[eid]

    def get_relation_id(self, relation):
        return self.relation2id[relation]

    def get_relation(self, rid):
        return self.id2relation[rid]

    def query(self, head, tail, relation):
        """
        Checks which facts are stored in the entire dataset.
        This method is overloaded for the task of link prediction,
        awaiting an incomplete statement and returning all known substitutes.

        :param head: Index of a head entity.
        :param tail: Index of a tail entity.
        :param relation: Index of a relation label.

        :return: A boolean array, deciding for each candidate whether or not the resulting statement is
            contained in the dataset.
        """
        raise NotImplementedError
        # if head is None:
        #     if tail is None:
        #         if relation is None:
        #             raise NotImplementedError('querying everything')
        #         raise NotImplementedError('querying full relation')
        #     if relation is None:
        #         raise NotImplementedError('querying full head')
        #     heads = np.zeros(self.shape[0], np.bool_)
        #     self.__library.query_head(get_array_pointer(heads), tail, relation)
        #     return heads
        # if tail is None:
        #     if relation is None:
        #         raise NotImplementedError('querying full tail')
        #     tails = np.zeros(self.shape[0], np.bool_)
        #     self.__library.query_tail(head, get_array_pointer(tails), relation)
        #     return tails
        # if relation is None:
        #     relations = np.zeros(self.shape[1], np.bool_)
        #     self.__library.query_rel(head, tail, get_array_pointer(relations))
        #     return relations
        # raise NotImplementedError('querying single facts')

    @staticmethod
    def read_benchmark(filename):
        with open(filename) as f:
            f.readline()  # Skip first line containing the number of rows
            triple_list = [(int(line.split()[0]), int(line.split()[1]), int(line.split()[2])) for line in f]
        return triple_list
