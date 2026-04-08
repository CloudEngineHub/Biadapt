CUDA_VISIBLE_DEVICES=3 python Ctpts/map_ctpts.py \
    --category Pen \
    --src_txt_path /home/zjinxian/Biadapt/stats/src_capping_Pen_list.txt \
    --src_data_dir /home/zjinxian/Biadapt/exp/Pen_Capping \
    --trg_img_dir /home/zjinxian/Biadapt/exp/Tmp_trg_img_dir/USB_capping \
    --out_dir /home/zjinxian/Biadapt/expdata/Pen2USB_capping \
    --prompt 'from a photo of a Pen to USB' \
    --ensemble_size 6