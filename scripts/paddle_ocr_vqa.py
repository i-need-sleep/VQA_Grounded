from paddleocr import PaddleOCR,draw_ocr

def main():
    ocr = PaddleOCR(use_angle_cls=True, lang='en', det_model_dir="../cache/ch_ppocr_server_v2.0_det_infer", cls_model_dir="../cache/ch_ppocr_mobile_v2.0_cls_infer/", rec_model_dir="../cache/ch_ppocr_server_v2.0_rec_infer/", vis_font_path='../cache/en_standard.ttf')
    img_path = '../datasets/VisWiz_VQA/train/VizWiz_train_00000011.jpg'
    result = ocr.ocr(img_path, cls=True)
    for line in result:
        print(line)


    # draw result
    from PIL import Image
    image = Image.open(img_path).convert('RGB')
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores, font_path='../cache/en_standard.ttf')
    im_show = Image.fromarray(im_show)
    im_show.save('../result.jpg')

if __name__ == '__main__':
    main()