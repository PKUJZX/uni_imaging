
# torchrun --standalone --nnodes=1 --nproc_per_node=8  ./main-ddp.py -cfg=config/simu.yaml -expName=simu

from model import SinoTx
import importlib.util
import torch, argparse, os, time, sys, shutil, yaml
from data import SinogramDataset
import numpy as np
import logging
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP


def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _load_sinogram_dataset_4npy_class():
    """加载 Unet_encdec_code_single/data.py，避免与本地包名 ``data`` 冲突。"""
    path = os.path.join(_repo_root(), 'Unet_encdec_code_single', 'data.py')
    if not os.path.isfile(path):
        raise FileNotFoundError(
            '未找到 Unet_encdec_code_single/data.py，请从 direct_inversion_usct 根目录运行或保持目录结构。'
        )
    spec = importlib.util.spec_from_file_location('unet_encdec_data', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SinogramDataset4npy


def _use_npy_dataset(params):
    ds = params.get('dataset') or {}
    ti = ds.get('train_input')
    if not ti:
        return False
    return os.path.isdir(os.path.expanduser(str(ti)))


def _masked_sinogram_rows(gt_ld, mask_row):
    """gt_ld: (L, D)；mask_row: (L,) 与 utils.random_masking 一致，1=该角度被 mask、未进 encoder。"""
    out = gt_ld.astype(np.float64).copy()
    m = np.asarray(mask_row).reshape(-1).astype(np.float64)
    if m.shape[0] != out.shape[0]:
        raise ValueError(f'mask 长度 {m.shape[0]} 与角度维 L={out.shape[0]} 不一致')
    for i in range(out.shape[0]):
        if m[i] > 0.5:
            out[i, :] = np.nan
    return out


def _save_mae_sinogram_comparison_png(save_path, gt_ld, pred_ld, mask_row):
    """
    三列：GT 正弦图 (角度×通道) | mask 后可见行（被移除行为 NaN 留白）| 重建。
    与主训练 ``comparison_ep*.png`` 类似，便于看 MAE 在输入序列上的效果。
    """
    masked_ld = _masked_sinogram_rows(gt_ld, mask_row)
    finite = np.concatenate(
        [gt_ld.reshape(-1), np.asarray(pred_ld, dtype=np.float64).reshape(-1)]
    )
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = float(np.percentile(finite, 1)), float(np.percentile(finite, 99))
        if vmax <= vmin:
            vmin, vmax = float(finite.min()), float(finite.max())
            if vmax <= vmin:
                vmin, vmax = 0.0, 1.0
    mse = float(np.mean((np.asarray(pred_ld, dtype=np.float64) - gt_ld.astype(np.float64)) ** 2))
    _kw = dict(cmap='jet', aspect='auto', vmin=vmin, vmax=vmax, origin='upper')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ('GT sinogram (angle × channel)', 'After mask (removed rows blank)', 'Reconstruction')
    for ax, arr, ti in zip(axes, (gt_ld, masked_ld, pred_ld), titles):
        im = ax.imshow(np.asarray(arr, dtype=np.float64), **_kw)
        ax.set_title(ti)
        ax.set_xlabel('channel')
        ax.set_ylabel('angle index')
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f'MAE sinogram | MSE={mse:.6g}', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

parser = argparse.ArgumentParser(description='SinoTx')
parser.add_argument('-gpus',   type=str, default="", help='list of visiable GPUs')
parser.add_argument('-expName',type=str, default="debug", help='Experiment name')
parser.add_argument('-cfg',    type=str, required=True, help='path to config yaml file')
parser.add_argument('-verbose',type=int, default=1, help='1:print to terminal; 0: redirect to file')

def main(args):
    # local_rank = int(os.environ["LOCAL_RANK"])
    # local_rank = int(os.environ.get("LOCAL_RANK", -1))
    # print(local_rank)
    # rank = int(os.environ["RANK"])
    # world_size = int(os.environ["WORLD_SIZE"])

    # if torch.distributed.is_nccl_available():
    #     print("yes") # yes
    #     torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    # else:
    #     print("no")
    #     torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)

    itr_out_dir = args.expName + '-itrOut'

    if os.path.isdir(itr_out_dir):
        shutil.rmtree(itr_out_dir)
    os.mkdir(itr_out_dir)
    # if rank == 0:
    #     if os.path.isdir(itr_out_dir):
    #         shutil.rmtree(itr_out_dir)
    #     os.mkdir(itr_out_dir) # to save temp output
    # torch.distributed.barrier()

    logging.basicConfig(filename=f"{args.expName}-itrOut/SinoTx.log", level=logging.DEBUG,\
                        format='%(asctime)s %(levelname)s %(module)s: %(message)s')
    if args.verbose:
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # training task init
    params = yaml.load(open(args.cfg, 'r'), Loader=yaml.CLoader)
    # logging.info(f"local rank {local_rank} (global rank {rank}) of a world size {world_size} started")
    # print(params)
    # torch.cuda.set_device(local_rank)

    logging.info("Init training dataset ...")
    if _use_npy_dataset(params):
        SinogramDataset4npy = _load_sinogram_dataset_4npy_class()
        train_ds = SinogramDataset4npy(params=params, is_train=True)
        valid_ds = SinogramDataset4npy(params=params, is_train=False)
        logging.info(
            'dataset: npy（与 Unet_encdec_code_single 一致：train_input / test_input + inorm/onorm）'
        )
        _shape = train_ds.in_shape
        _seqlen = train_ds.in_seqlen
        _cdim = train_ds.in_cdim
    else:
        ds_cfg = params.get('dataset') or {}
        if not ds_cfg.get('th5') or not ds_cfg.get('vh5'):
            raise ValueError(
                '请在 yaml 的 dataset 中配置 train_input（npy 目录）或同时配置 th5、vh5（HDF5）'
            )
        train_ds = SinogramDataset(ifn=ds_cfg['th5'], params=params)
        valid_ds = SinogramDataset(ifn=ds_cfg['vh5'], params=params)
        logging.info('dataset: HDF5（th5 / vh5）')
        _shape = train_ds.shape
        _seqlen = train_ds.seqlen
        _cdim = train_ds.cdim

    train_dl = DataLoader(
        train_ds,
        batch_size=params['train']['mbsz'],
        num_workers=4,
        prefetch_factor=params['train']['mbsz'],
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )
    logging.info('%d samples, in_shape=%s (seqlen=%s cdim=%s), training' % (len(train_ds), _shape, _seqlen, _cdim))

    valid_dl = DataLoader(
        valid_ds,
        batch_size=params['train']['mbsz'],
        num_workers=4,
        prefetch_factor=params['train']['mbsz'],
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )
    logging.info('%d samples, validation' % (len(valid_ds),))

    model = SinoTx(seqlen=_seqlen, in_dim=_cdim, params=params).cuda()
    # model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=params['train']['lr'], betas=(0.9, 0.95))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.4, verbose=0)

    save_dir = os.path.join(itr_out_dir, 'image')
    os.makedirs(save_dir, exist_ok=True)
    viz_every_n_epochs = int((params.get('train') or {}).get('viz_every_n_epochs', 1))

    logging.info(
        'Start training ... (MAE sinogram comparison png every %s epochs under image/, 0=off)',
        viz_every_n_epochs,
    )
    for ep in range(1, params['train']['maxep']+1):
        # optimizer.step()
        train_ep_tick = time.time()
        # print(params['train']['maxep'])
        # print(train_dl[0].shape)
        for batch in train_dl:
            imgs_tr = batch[0] if isinstance(batch, (list, tuple)) else batch
            optimizer.zero_grad()
            loss, pred, mask = model.forward(imgs_tr.cuda())
            loss.backward()
            optimizer.step()

        lr_scheduler.step()
        # if rank != world_size-1: continue

        time_e2e = time.time() - train_ep_tick
        itr_prints = '[Train] Epoch %3d, loss: %.6f, elapse: %.2fs/epoch, %d steps with lbs=%d' % (\
                     ep, loss.cpu().detach().numpy(), time_e2e, len(train_dl), imgs_tr.shape[0])
        logging.info(itr_prints)

        val_loss = []
        valid_ep_tick = time.time()
        for batch in valid_dl:
            imgs_val = batch[0] if isinstance(batch, (list, tuple)) else batch
            with torch.no_grad():
                _vloss, _vpred, _vmask = model.forward(imgs_val.cuda())
                val_loss.append(_vloss.cpu().numpy())

        valid_e2e = time.time() - valid_ep_tick
        _prints = '[Valid] Epoch %3d, loss: %.6f, elapse: %.2fs/epoch\n' % (ep, np.mean(val_loss), valid_e2e)
        logging.info(_prints)

        if (
            viz_every_n_epochs > 0
            and ep % viz_every_n_epochs == 0
            and len(train_dl) > 0
        ):
            bi = min(1, imgs_tr.shape[0] - 1)
            gt = imgs_tr[bi, 0].detach().cpu().numpy()
            pr = pred[bi].detach().cpu().numpy()
            mk = mask[bi].detach().cpu().numpy()
            png_path = os.path.join(save_dir, f'mae_sinogram_comparison_ep{ep:05d}.png')
            _save_mae_sinogram_comparison_png(png_path, gt, pr, mk)
            logging.info('saved MAE viz %s', png_path)
            if len(valid_dl) > 0:
                bi_v = min(1, imgs_val.shape[0] - 1)
                gt_v = imgs_val[bi_v, 0].detach().cpu().numpy()
                pr_v = _vpred[bi_v].detach().cpu().numpy()
                mk_v = _vmask[bi_v].detach().cpu().numpy()
                png_v = os.path.join(save_dir, f'mae_sinogram_comparison_valid_ep{ep:05d}.png')
                _save_mae_sinogram_comparison_png(png_v, gt_v, pr_v, mk_v)
                logging.info('saved MAE viz %s', png_v)

        if ep % params['train']['ckp_steps'] != 0: continue


        # general version
        torch.save(model.state_dict(), "%s/mdl-ep%05d.pth" % (itr_out_dir, ep))
        torch.save(model.encoder.state_dict(), "%s/encoder-ep%05d.pth" % (itr_out_dir, ep))
        torch.save(model.decoder.state_dict(), "%s/decoder-ep%05d.pth" % (itr_out_dir, ep))
        
        # torch.jit.save(torch.jit.trace(model, (imgs_tr[:1].cuda(), 0.7)), "%s/script-ep%05d.pth" % (itr_out_dir, ep))
        with open(f'{itr_out_dir}/config.yaml', 'w') as fp:
            yaml.dump(params, fp)


if __name__ == "__main__":
    args, unparsed = parser.parse_known_args()

    print(args,unparsed)
    if len(unparsed) > 0:
        print('Unrecognized argument(s): \n%s \nProgram exiting ... ... ' % '\n'.join(unparsed))
        exit(0)

    if len(args.gpus) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    main(args)
