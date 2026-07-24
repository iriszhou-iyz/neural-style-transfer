# neural-style-transfer
experimenting with neural style transfer on my own photos!

**neural style transfer**: the ability to create a new image based on two input images: one for content and the other for artistic style. the output image thus looks like the content image created in the style of the style reference image.

this is a lightweight PyTorch implementation of art style transfer using a pre-trained VGG19 model, as discussed in [Gatys et al. 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) 

## usage
```
    git clone https://github.com/iriszhou-iyz/neural-style-transfer.git
    cd neural-style-transfer
    pip install -r requirements.txt
```

## how it works
1. **load and normalize the content and style images.** 

    images are resized and normalized using ImageNet statistics so they match the input distribution expected by the pretrained VGG19 network. 

2. **extract deep features using a pretrained, frozen VGG19 network.** 

    **content representation:** we use the **conv4_2** layer, which is sufficiently deep to capture the high-level semantic structure of the image (object shapes, spatial layout, and overall structre) while ignoring low-level pixel details such as color and texture. 

    **style representation:** we use the **conv1_1, conv2_1, conv3_1, conv4_1, and conv5_1** layers. combining multiple layers captures artistic characteristics at different scales: low-level properties (colors, brush strokes, edges) in early layers to higher-level textures and larger patterns in deeper layers. 

3. **initialize the generated image.** 

    the generated image is initialized as a copy of the content image, then iteratively modified during optimization.

4. **compute content loss.** 

    compare the feature representations of the generated and content images at the **conv4_2** layer to preserve the original image structure.

5. **compute style loss.** 

    compare the **Gram matrices** of the generated and style image feature maps across the selected VGG layers to match texture and artistic style while ignoring spatial arrangement.

6. **optimize the generated image.** 

    use gradient descent to minimize the weighted sum of the content and style losses, updating only the pixels of the generated image.