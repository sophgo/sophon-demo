import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import product
from math import sqrt
from typing import List
from collections import defaultdict
from config import cfg, mask_type
from backbone import construct_backbone

# This is required for Pytorch 1.0.1 on Windows to initialize Cuda on some driver versions.
# See the bug report here: https://github.com/pytorch/pytorch/issues/17108
device = 'cpu'
if torch.cuda.is_available():
	torch.cuda.current_device()
	device = 'cuda'

# As of March 10, 2019, Pytorch DataParallel still doesn't support JIT Script Modules
# use_jit = torch.cuda.device_count() <= 1
# if not use_jit:
#     print('Multiple GPUs detected! Turning off JIT.')
#
# ScriptModuleWrapper = torch.jit.ScriptModule if use_jit else nn.Module
# script_method_wrapper = torch.jit.script_method if use_jit else lambda fn, _rcn=None: fn

class Concat(nn.Module):
	def __init__(self, nets, extra_params):
		super().__init__()

		self.nets = nn.ModuleList(nets)
		self.extra_params = extra_params

	def forward(self, x):
		# Concat each along the channel dimension
		return torch.cat([net(x) for net in self.nets], dim=1, **self.extra_params)

class InterpolateModule(nn.Module):
	"""
	This is a module version of F.interpolate (rip nn.Upsampling).
	Any arguments you give it just get passed along for the ride.
	"""

	def __init__(self, *args, **kwdargs):
		super().__init__()

		self.args = args
		self.kwdargs = kwdargs

	def forward(self, x):
		# return F.interpolate(x, *self.args, **self.kwdargs)
		return F.interpolate(x, size=(int(x.shape[2] * self.kwdargs['scale_factor']), int(x.shape[3] * self.kwdargs['scale_factor'])), mode=self.kwdargs['mode'], align_corners=self.kwdargs['align_corners'])

def make_net(in_channels, conf, include_last_relu=True):
	def make_layer(layer_cfg):
		nonlocal in_channels

		# Possible patterns:
		# ( 256, 3, {}) -> conv
		# ( 256,-2, {}) -> deconv
		# (None,-2, {}) -> bilinear interpolate
		# ('cat',[],{}) -> concat the subnetworks in the list
		#
		# You know it would have probably been simpler just to adopt a 'c' 'd' 'u' naming scheme.
		# Whatever, it's too late now.
		if isinstance(layer_cfg[0], str):
			layer_name = layer_cfg[0]

			if layer_name == 'cat':
				nets = [make_net(in_channels, x) for x in layer_cfg[1]]
				layer = Concat([net[0] for net in nets], layer_cfg[2])
				num_channels = sum([net[1] for net in nets])
		else:
			num_channels = layer_cfg[0]
			kernel_size = layer_cfg[1]

			if kernel_size > 0:
				layer = nn.Conv2d(in_channels, num_channels, kernel_size, **layer_cfg[2])
			else:
				if num_channels is None:
					layer = InterpolateModule(scale_factor=-kernel_size, mode='bilinear', align_corners=False, **layer_cfg[2])
					# layer = nn.Upsample(scale_factor=-kernel_size, mode='bilinear', align_corners=False)
				else:
					layer = nn.ConvTranspose2d(in_channels, num_channels, -kernel_size, **layer_cfg[2])

		in_channels = num_channels if num_channels is not None else in_channels

		# Don't return a ReLU layer if we're doing an upsample. This probably doesn't affect anything
		# output-wise, but there's no need to go through a ReLU here.
		# Commented out for backwards compatibility with previous models
		# if num_channels is None:
		#     return [layer]
		# else:
		return [layer, nn.ReLU(inplace=True)]

	# Use sum to concat together all the component layer lists
	net = sum([make_layer(x) for x in conf], [])
	if not include_last_relu:
		net = net[:-1]

	return nn.Sequential(*(net)), in_channels

prior_cache = defaultdict(lambda: None)

class PredictionModule(nn.Module):
	def __init__(self, in_channels, out_channels=1024, aspect_ratios=[[1]], scales=[1], parent=None, index=0):
		super().__init__()

		self.num_classes = cfg.num_classes
		self.mask_dim    = cfg.mask_dim # Defined by Yolact
		self.num_priors  = sum(len(x)*len(scales) for x in aspect_ratios)
		self.parent      = [parent] # Don't include this in the state dict
		self.index       = index
		self.num_heads   = cfg.num_heads # Defined by Yolact

		if parent is None:
			self.upfeature, out_channels = make_net(in_channels, cfg.extra_head_net)

			self.bbox_layer = nn.Conv2d(out_channels, self.num_priors * 4,                **cfg.head_layer_params)
			self.conf_layer = nn.Conv2d(out_channels, self.num_priors * self.num_classes, **cfg.head_layer_params)
			self.mask_layer = nn.Conv2d(out_channels, self.num_priors * self.mask_dim,    **cfg.head_layer_params)

			# What is this ugly lambda doing in the middle of all this clean prediction module code?
			def make_extra(num_layers):
				if num_layers == 0:
					return lambda x: x
				else:
					# Looks more complicated than it is. This just creates an array of num_layers alternating conv-relu
					return nn.Sequential(*sum([[
						nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
						nn.ReLU(inplace=True)
					] for _ in range(num_layers)], []))

			self.bbox_extra, self.conf_extra, self.mask_extra = [make_extra(x) for x in cfg.extra_layers]

		self.aspect_ratios = aspect_ratios
		self.scales = scales

		self.priors = None
		self.last_conv_size = None
		self.last_img_size = None

	def forward(self, x):
		src = self if self.parent[0] is None else self.parent[0]
		x = src.upfeature(x)

		bbox_x = src.bbox_extra(x)
		conf_x = src.conf_extra(x)
		mask_x = src.mask_extra(x)

		bbox = src.bbox_layer(bbox_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
		conf = src.conf_layer(conf_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)

		mask = src.mask_layer(mask_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_dim)

		mask = torch.tanh(mask)

		# conv_h = x.size(2)
		# conv_w = x.size(3)
		# priors = self.make_priors(conv_h, conv_w, x.device)
		# preds = { 'loc': bbox, 'conf': conf, 'mask': mask, 'priors': priors }
		# return preds
		return bbox, conf, mask

	def make_priors(self, conv_h, conv_w, device):
		""" Note that priors are [x,y,width,height] where (x,y) is the center of the box. """
		global prior_cache
		size = (conv_h, conv_w)

		if self.last_img_size != (cfg._tmp_img_w, cfg._tmp_img_h):
			prior_data = []

			# Iteration order is important (it has to sync up with the convout)
			for j, i in product(range(conv_h), range(conv_w)):
				# +0.5 because priors are in center-size notation
				x = (i + 0.5) / conv_w
				y = (j + 0.5) / conv_h

				for ars in self.aspect_ratios:
					for scale in self.scales:
						for ar in ars:
							if not cfg.backbone.preapply_sqrt:
								ar = sqrt(ar)

							if cfg.backbone.use_pixel_scales:
								w = scale * ar / cfg.max_size
								h = scale / ar / cfg.max_size
							else:
								w = scale * ar / conv_w
								h = scale / ar / conv_h

							# This is for backward compatability with a bug where I made everything square by accident
							if cfg.backbone.use_square_anchors:
								h = w

							prior_data += [x, y, w, h]

			self.priors = torch.Tensor(prior_data).view(-1, 4).detach().to(device)
			# self.priors = torch.Tensor(prior_data).view(-1, 4).detach()
			self.priors.requires_grad = False
			self.last_img_size = (cfg._tmp_img_w, cfg._tmp_img_h)
			self.last_conv_size = (conv_w, conv_h)
			prior_cache[size] = None
		elif self.priors.device != device:
			# This whole weird situation is so that DataParalell doesn't copy the priors each iteration
			if prior_cache[size] is None:
				prior_cache[size] = {}

			if device not in prior_cache[size]:
				prior_cache[size][device] = self.priors.to(device)

			self.priors = prior_cache[size][device]

		return self.priors

class FPN(nn.Module):
	"""
	Implements a general version of the FPN introduced in
	https://arxiv.org/pdf/1612.03144.pdf

	Parameters (in cfg.fpn):
		- num_features (int): The number of output features in the fpn layers.
		- interpolation_mode (str): The mode to pass to F.interpolate.
		- num_downsample (int): The number of downsampled layers to add onto the selected layers.
								These extra layers are downsampled from the last selected layer.

	Args:
		- in_channels (list): For each conv layer you supply in the forward pass,
							  how many features will it have?
	"""
	__constants__ = ['interpolation_mode', 'num_downsample', 'use_conv_downsample', 'relu_pred_layers',
					 'lat_layers', 'pred_layers', 'downsample_layers', 'relu_downsample_layers']

	def __init__(self, in_channels):
		super().__init__()

		self.lat_layers  = nn.ModuleList([
			nn.Conv2d(x, cfg.fpn.num_features, kernel_size=1)
			for x in reversed(in_channels)
		])

		# This is here for backwards compatability
		padding = 1 if cfg.fpn.pad else 0
		self.pred_layers = nn.ModuleList([
			nn.Conv2d(cfg.fpn.num_features, cfg.fpn.num_features, kernel_size=3, padding=padding)
			for _ in in_channels
		])

		if cfg.fpn.use_conv_downsample:
			self.downsample_layers = nn.ModuleList([
				nn.Conv2d(cfg.fpn.num_features, cfg.fpn.num_features, kernel_size=3, padding=1, stride=2)
				for _ in range(cfg.fpn.num_downsample)
			])

		self.interpolation_mode     = cfg.fpn.interpolation_mode
		self.num_downsample         = cfg.fpn.num_downsample
		self.use_conv_downsample    = cfg.fpn.use_conv_downsample
		self.relu_downsample_layers = cfg.fpn.relu_downsample_layers
		self.relu_pred_layers       = cfg.fpn.relu_pred_layers

	# @script_method_wrapper
	def forward(self, convouts:List[torch.Tensor]):
		"""
		Args:
			- convouts (list): A list of convouts for the corresponding layers in in_channels.
		Returns:
			- A list of FPN convouts in the same order as x with extra downsample layers if requested.
		"""

		out = []
		x = torch.zeros(1, device=convouts[0].device)
		for i in range(len(convouts)):
			out.append(x)

		# For backward compatability, the conv layers are stored in reverse but the input and output is
		# given in the correct order. Thus, use j=-i-1 for the input and output and i for the conv layers.
		j = len(convouts)
		for lat_layer in self.lat_layers:
			j -= 1

			if j < len(convouts) - 1:
				x = F.interpolate(x, size=(int(convouts[j].shape[2]), int(convouts[j].shape[3])), mode=self.interpolation_mode, align_corners=False)

			x = x + lat_layer(convouts[j])
			out[j] = x

		# This janky second loop is here because TorchScript.
		j = len(convouts)
		for pred_layer in self.pred_layers:
			j -= 1
			out[j] = pred_layer(out[j])

			if self.relu_pred_layers:
				F.relu(out[j], inplace=True)

		cur_idx = len(out)

		# In the original paper, this takes care of P6
		if self.use_conv_downsample:
			for downsample_layer in self.downsample_layers:
				out.append(downsample_layer(out[-1]))
		else:
			for idx in range(self.num_downsample):
				# Note: this is an untested alternative to out.append(out[-1][:, :, ::2, ::2]). Thanks TorchScript.
				out.append(nn.functional.max_pool2d(out[-1], 1, stride=2))

		if self.relu_downsample_layers:
			for idx in range(len(out) - cur_idx):
				out[idx] = F.relu(out[idx + cur_idx], inplace=False)

		return out

class FastMaskIoUNet(nn.Module):

	def __init__(self):
		super().__init__()
		input_channels = 1
		last_layer = [(cfg.num_classes-1, 1, {})]
		self.maskiou_net, _ = make_net(input_channels, cfg.maskiou_net + last_layer, include_last_relu=True)

	def forward(self, x):
		x = self.maskiou_net(x)
		maskiou_p = F.max_pool2d(x, kernel_size=x.size()[2:]).squeeze(-1).squeeze(-1)

		return maskiou_p



class Yolact(nn.Module):
	"""


	██╗   ██╗ ██████╗ ██╗      █████╗  ██████╗████████╗
	╚██╗ ██╔╝██╔═══██╗██║     ██╔══██╗██╔════╝╚══██╔══╝
	 ╚████╔╝ ██║   ██║██║     ███████║██║        ██║
	  ╚██╔╝  ██║   ██║██║     ██╔══██║██║        ██║
	   ██║   ╚██████╔╝███████╗██║  ██║╚██████╗   ██║
	   ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝   ╚═╝


	You can set the arguments by changing them in the backbone config object in config.py.

	Parameters (in cfg.backbone):
		- selected_layers: The indices of the conv layers to use for prediction.
		- pred_scales:     A list with len(selected_layers) containing tuples of scales (see PredictionModule)
		- pred_aspect_ratios: A list of lists of aspect ratios with len(selected_layers) (see PredictionModule)
	"""

	def __init__(self):
		super().__init__()

		self.backbone = construct_backbone(cfg.backbone)
		# Compute mask_dim here and add it back to the config. Make sure Yolact's constructor is called early!
		if cfg.mask_type == mask_type.direct:
			cfg.mask_dim = cfg.mask_size**2
		elif cfg.mask_type == mask_type.lincomb:
			if cfg.mask_proto_use_grid:
				self.grid = torch.Tensor(np.load(cfg.mask_proto_grid_file))
				self.num_grids = self.grid.size(0)
			else:
				self.num_grids = 0

			self.proto_src = cfg.mask_proto_src

			if self.proto_src is None: in_channels = 3
			elif cfg.fpn is not None: in_channels = cfg.fpn.num_features
			else: in_channels = self.backbone.channels[self.proto_src]
			in_channels += self.num_grids

			# The include_last_relu=false here is because we might want to change it to another function
			self.proto_net, cfg.mask_dim = make_net(in_channels, cfg.mask_proto_net, include_last_relu=False)

			if cfg.mask_proto_bias:
				cfg.mask_dim += 1


		self.selected_layers = cfg.backbone.selected_layers
		src_channels = self.backbone.channels

		if cfg.use_maskiou:
			self.maskiou_net = FastMaskIoUNet()

		# if cfg.fpn is not None:
		#     # Some hacky rewiring to accomodate the FPN
		#     self.fpn = FPN([src_channels[i] for i in self.selected_layers])
		#     self.selected_layers = list(range(len(self.selected_layers) + cfg.fpn.num_downsample))
		#     src_channels = [cfg.fpn.num_features] * len(self.selected_layers)
		self.fpn = FPN([src_channels[i] for i in self.selected_layers])
		self.selected_layers = list(range(len(self.selected_layers) + cfg.fpn.num_downsample))
		src_channels = [cfg.fpn.num_features] * len(self.selected_layers)

		self.prediction_layers = nn.ModuleList()
		cfg.num_heads = len(self.selected_layers)

		for idx, layer_idx in enumerate(self.selected_layers):
			# If we're sharing prediction module weights, have every module's parent be the first one
			parent = None
			if cfg.share_prediction_module and idx > 0:
				parent = self.prediction_layers[0]

			pred = PredictionModule(src_channels[layer_idx], src_channels[layer_idx],
									aspect_ratios = cfg.backbone.pred_aspect_ratios[idx],
									scales        = cfg.backbone.pred_scales[idx],
									parent        = parent,
									index         = idx)
			self.prediction_layers.append(pred)

		# Extra parameters for the extra losses
		if cfg.use_class_existence_loss:
			# This comes from the smallest layer selected
			# Also note that cfg.num_classes includes background
			self.class_existence_fc = nn.Linear(src_channels[-1], cfg.num_classes - 1)

		if cfg.use_semantic_segmentation_loss:
			self.semantic_seg_conv = nn.Conv2d(src_channels[0], cfg.num_classes-1, kernel_size=1)

		# # For use in evaluation
		# self.detect = Detect(cfg.num_classes, bkg_label=0, top_k=cfg.nms_top_k,
		#     conf_thresh=cfg.nms_conf_thresh, nms_thresh=cfg.nms_thresh)

	def load_weights(self, path):
		""" Loads weights from a compressed save file. """
		state_dict = torch.load(path, map_location=device)

		# For backward compatability, remove these (the new variable is called layers)
		for key in list(state_dict.keys()):
			if key.startswith('backbone.layer') and not key.startswith('backbone.layers'):
				del state_dict[key]

			# Also for backward compatibility with v1.0 weights, do this check
			if key.startswith('fpn.downsample_layers.'):
				if cfg.fpn is not None and int(key.split('.')[2]) >= cfg.fpn.num_downsample:
					del state_dict[key]
		self.load_state_dict(state_dict)

	def forward(self, x):
		""" The input should be of size [batch_size, 3, img_h, img_w] """
		_, _, img_h, img_w = x.size()
		cfg._tmp_img_h = img_h
		cfg._tmp_img_w = img_w

		outs = self.backbone(x)

		outs = [outs[i] for i in cfg.backbone.selected_layers]
		outs = self.fpn(outs)
		print('fpn shape    > ', [i.shape[-1] for i in outs])
		proto_x = x if self.proto_src is None else outs[self.proto_src]

		if self.num_grids > 0:
			grids = self.grid.repeat(proto_x.size(0), 1, 1, 1)
			proto_x = torch.cat([proto_x, grids], dim=1)

		proto_out = self.proto_net(proto_x)
		proto_out = F.relu(proto_out)

		# Move the features last so the multiplication is easy
		proto_out = proto_out.permute(0, 2, 3, 1).contiguous()
		loc, conf, mask = [], [], []
		for idx, pred_layer in zip(self.selected_layers, self.prediction_layers):
			pred_x = outs[idx]
			# A hack for the way dataparallel works
			if cfg.share_prediction_module and pred_layer is not self.prediction_layers[0]:
				pred_layer.parent = [self.prediction_layers[0]]

			p = pred_layer(pred_x)    ###loc, conf, mask
			loc.append(p[0])
			conf.append(p[1])
			mask.append(p[2])
		loc = torch.cat(loc, -2)
		conf = torch.cat(conf, -2)
		mask = torch.cat(mask, -2)
		conf = F.softmax(conf, -1)

		print('output shape    > ', loc.shape, conf.shape, mask.shape, proto_out.shape)

		return loc, conf, mask, proto_out
