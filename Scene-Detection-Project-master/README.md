# Repro-VAE Scene Detection in Anime
This is the course project of 11785 CMU, named Repro-VAE Scene Detection in Anime
 
Team Members: Dijing Zhang, Siqiao Fu, Yiping Dong, Zhengyang Zou

### Abstract
2D animation has become a popular category in the field of films. With the help of digital tech, artists are able to comprise great masterpieces in fairly short amount of time. Beyond the fine scenery, the meaning inside every frame is vastly explored for various reasons. For example, scene classification is one of the most explored application when it comes to animation, which can facilitate a wide-range of tasks like anime to text, cross anime scene retrieval and human-centric storyline construction. There have been some supervised learning methods that tackle the problem of scene change detection in animation. However, they all require painful pre-processing to get the ground truth of scene change labels, which can only be acquired by long and tedious manual work. Our goal in this article is to come up with an unsupervised model and get the job done. The significant contribution will lie in the elimination of labeling the whole anime as pre-processing. 

We use VAE as the baseline model. Although VAE may not seem anything related to the topic of scene change detection, it helps compress the image into a lower but more informative latent space as the representation of that certain image. This compressed representation significantly helps in the task of scene change detection, which will be illustrated in the following acticle. Beyond that, we explored a novel training method to regularize the latent space and we call it "reprojection error". During experiments we found that it did improve the accuracy in most cases. The main contribution:1) achieved reasonable scene change accuracy without the help of labeled training dataset; 2) explored VAE with "reprojection" regularization term, the repro-VAE we create seems better in the field of representative learning.

### Dataset Your Name (Kimi no Na wa)

<center><img src="https://github.com/11785-Team/Scene-Detection-Project/blob/master/images/example_images.png" alt=""></center>

We use 142p as our dataset in the project. The source image size is 189×142, which will be re-scaled into 64×64.

### Defination of Scene Change

We define scene change as there exists a scene change in the image, like the whole background changes from forest to city, or from home to classroom. Or a great change of POV. The main character or certain objects change doesn't lead to a scene change. Here is a example.

<center><img src="https://github.com/11785-Team/Scene-Detection-Project/blob/master/images/Scene_change_exp.png" alt=""></center>

### Architecture and Model
Based on β-VAE architecture and introduce our innovation idea: reprojection loss. Create one VAE named "repro-VAE". refer to [*Anand Krishnamoorthy, PyTorch-VAE, (2020), GitHub repository, https://github.com/AntixK/PyTorch-VAE/tree/master/models*]

**Here is the architecture of our model**  (*refer to Hung-yi Lee’s lecture  https://www.youtube.com/watch?v=0CKeqXl5IY0&t=1650s*)

<center><img src="https://github.com/11785-Team/Scene-Detection-Project/blob/master/images/architecture.png" alt=""></center>
<center><img src="https://github.com/11785-Team/Scene-Detection-Project/blob/master/images/3D_architecture.png" alt=""></center>

**Here is the visualization explanation of reprojection loss.**

<div  align="center"><center><img src="https://github.com/11785-Team/Scene-Detection-Project/blob/master/images/reprojection.png" alt=""></center></div>
<div  align="center"><center><img src="https://github.com/11785-Team/Scene-Detection-Project/blob/master/images/latent_reprojection.png" alt="" width="1032px", height="360px"></center></div>

**Here is the detailed architecture**

<div  align="center"><center><img src="https://github.com/11785-Team/Scene-Detection-Project/blob/master/images/architecture_details.png" alt=""></center></div>

### Results
This is one example of reconstruction images. Because what we want is to detect the scene change instead of reconstruction images, the image quality is not very good but we can still tell the basic frame.

<div  align="center"><center><img src="https://github.com/11785-Team/Scene-Detection-Project/blob/master/images/reconstruction.png" alt=""></center></div>

These are the evluation of our baseline model and repro-VAE model about the accuracy of scene change detection

**Baseline model**

<div  align="center"><center><img src="https://github.com/11785-Team/Scene-Detection-Project/blob/master/images/baseline_result.png" alt="" width="800px", height="200px"></center></div>

**Repro-VAE model**

<div  align="center"><center><img src="https://github.com/11785-Team/Scene-Detection-Project/blob/master/images/model_result.png" alt="" width="800px", height="200px"></center></div>

As far as we can see, our repro-VAE has promising increase in the accuracy compared with the baseline VAE model. Its further potential needs exploration.

To be clear, we also visualize the latent space by using t-SNE. Here is the illustration.

<div  align="center"><center><img src="https://github.com/11785-Team/Scene-Detection-Project/blob/master/images/TSNE-latentspcae.png" alt=""></center></div>

As we can see, the consective images without scene changes are clustered into separate groups.

**Here are some visualizaiton of scene change detection**

**True Positive**

<div  align="center"><center><img src="https://github.com/11785-Team/Scene-Detection-Project/blob/master/images/True_Positive.png" alt=""></center></div>

**Dynamic changes-False Negative**

<div  align="center"><center><img src="https://github.com/11785-Team/Scene-Detection-Project/blob/master/images/dynamic_change.png" alt=""></center></div>

**True Negative**

<div  align="center"><center><img src="https://github.com/11785-Team/Scene-Detection-Project/blob/master/images/True_Negative.png" alt=""></center></div>

**False Postive**

<div  align="center"><center><img src="https://github.com/11785-Team/Scene-Detection-Project/blob/master/images/False_Positive.png" alt=""></center></div>


### Video presentation
Youtube link: https://www.youtube.com/watch?v=8YoGIvvyqGs&list=PLp-0K3kfddPw7yEP_cICv9Glt237KNpSx&index=17

## How to train the model and do the inference
Platform: Colab

Opts.py stores all the hyperparameters that you can refer to. Here is the hyperparameters we are using to get the best performance:

**Do training**:

    !python main.py --val_folder [Your validation dataset folder] --train_folder [Your training dataset folder] --bs 256  --hidden-dims [32, 64, 128, 256] --max_iters 100 --loss_type H --lr 0.0001 --latent_dim 10 --tau 200 --beta 4 --output_folder [Your result folder]

**Do inference**:

    !python inference.py --beta 4 --latent_dim 10 --bs 256 --span 1 --image_folder [Your validation dataset folder]  --model_folder [Your model_state folder]

**Get Acc**:

    !python pure_test.py --labels [Your labels file] --dictionary [Your result npy folder]
