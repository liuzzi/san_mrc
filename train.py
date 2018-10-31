import re
import os
import sys
import random
import string
import logging
import argparse
import json
import torch
import msgpack
import pandas as pd
import numpy as np
from shutil import copyfile
from datetime import datetime
from collections import Counter, defaultdict
from src.model import DocReaderModel
from src.batcher import load_meta, BatchGen
from config import set_args
from my_utils.utils import set_environment
from my_utils.log_wrapper import create_logger
from my_utils.squad_eval import evaluate
from my_utils.data_utils import predict_squad, gen_name, load_squad_v2_label, compute_acc
from my_utils.squad_eval_v2 import my_evaluation as evaluate_v2

args = set_args()
# set model dir
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)
model_dir = os.path.abspath(model_dir)

# set environment
set_environment(args.seed, args.cuda)
# setup logger
logger =  create_logger(__name__, to_disk=True, log_file=args.log_file)

def load_squad(data_path):
	with open(data_path) as dataset_file:
		dataset_json = json.load(dataset_file)
		dataset = dataset_json['data']
		return dataset

def main():
	opt = vars(args)
	logger.info('Loading Squad')
	version = 'v1'
	if args.v2_on:
		version = 'v2'
		dev_labels = load_squad_v2_label(args.dev_gold)

	logger.info('Loading Meta')
	embedding, opt = load_meta(opt, gen_name(args.data_dir, args.meta, version, suffix='pick'))
	
	logger.info('Loading Train Batcher')
	train_data = BatchGen(gen_name(args.data_dir, args.train_data, version),
						  batch_size=args.batch_size,
						  gpu=args.cuda,
						  with_label=args.v2_on)

	logger.info('Loading Test Batcher')
	dev_data = BatchGen(gen_name(args.data_dir, args.dev_data, version),
						  batch_size=args.batch_size,
						  gpu=args.cuda, is_train=False)

	logger.info('Loading Golden Standards')
	# load golden standard
	dev_gold = load_squad(args.dev_gold)

	if len(args.resume) > 0:
		logger.info('Loading resumed model')
		model = DocReaderModel.load(args.resume,embedding,gpu=args.cuda)
		resumeSplit = args.resume.split('_')

		best_f1_score = float(resumeSplit[6].replace('.pt',''))
		best_em_score = float(resumeSplit[4])
		resumed_epoch = int(resumeSplit[2]) + 1

		#step scheduler
		for i in range(resumed_epoch):
			model.scheduler.step()

		logger.info("RESUMING MODEL TRAINING. BEST epoch {} EM {} F1 {} ".format(str(resumed_epoch),str(best_em_score),str(best_f1_score)))

	else:
		model = DocReaderModel(opt, embedding)
		best_em_score, best_f1_score = 0.0, 0.0
		resumed_epoch = 0

	# model meta str
	# headline = '############# Model Arch of SAN #############'
	# print network
	# logger.info('\n{}\n{}\n'.format(headline, model.network))
	model.setup_eval_embed(embedding)

	logger.info("Total number of params: {}".format(model.total_param))
	logger.info("Batch Size {}".format(args.batch_size))
	if args.cuda:
		model.cuda()
	else:
		model.cpu()
	
	for epoch in range(resumed_epoch, args.epoches):
		logger.warning('At epoch {}'.format(epoch))

		#shuffle training batch 
		train_data.reset()
		start = datetime.now()
		for i, batch in enumerate(train_data):
			model.update(batch)
			if (model.updates) % args.log_per_updates == 0 or i == 0:
				logger.info('#updates[{0:6}] train loss[{1:.5f}] remaining[{2}]'.format(
					model.updates, model.train_loss.avg,
					str((datetime.now() - start) / (i + 1) * (len(train_data) - i - 1)).split('.')[0]))
		# dev eval
		results, labels = predict_squad(model, dev_data, v2_on=args.v2_on)
		if args.v2_on:
			metric = evaluate_v2(dev_gold, results, na_prob_thresh=args.classifier_threshold)
			em, f1 = metric['exact'], metric['f1']
			acc = compute_acc(labels, dev_labels)
		else:
			metric = evaluate(dev_gold, results)
			em, f1 = metric['exact_match'], metric['f1']

		output_path = os.path.join(model_dir, 'dev_output_{}.json'.format(epoch))
		with open(output_path, 'w') as f:
			json.dump(results, f)

		# setting up scheduler
		if model.scheduler is not None:
			logger.info('scheduler_type {}'.format(opt['scheduler_type']))
			if opt['scheduler_type'] == 'rop':
				model.scheduler.step(f1, epoch=epoch)
			else:
				model.scheduler.step()
		# save
		model_file = os.path.join(model_dir, 'cp_epoch_{}_em_{}_f1_{}.pt'.format( epoch, int(em), int(f1)))

		model.save(model_file, epoch)
		if em + f1 > best_em_score + best_f1_score:
			copyfile(os.path.join(model_dir, model_file), os.path.join(model_dir, 'best_epoch_{}_em_{}_f1_{}.pt'.format(epoch, int(em), int(f1))))
			best_em_score, best_f1_score = em, f1
			logger.info('Saved the new best model and prediction')

		logger.warning("Epoch {0} - dev EM: {1:.3f} F1: {2:.3f} (best EM: {3:.3f} F1: {4:.3f})".format(epoch, em, f1, best_em_score, best_f1_score))
		if args.v2_on:
			logger.warning("Epoch {0} - ACC: {1:.4f}".format(epoch, acc))
		if metric is not None:
			logger.warning("Detailed Metric at Epoch {0}: {1}".format(epoch, metric))

if __name__ == '__main__':
	main()
