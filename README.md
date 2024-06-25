# ChromaFlow

ComfyUI ChromaFlow creates smoothly Animated Masks with synchronized Depth Maps from an input image. It introduces the concept of a "motion guidance" image, and is great for doing complex motion with interacting elements for AnimateDiff and IPAdapter workflows.

ChromaFlow is currently in Beta and we are definitely still working out some issues we already know of which will be listed below.  There is still a lot of exploration to be done because it is essentially pulling the spatio-temporal AnimateDiff model's space around forcing it to comply to very abstract movement, and generate according to the mask.

## Installation
1. Clone this repository
2. `cd` to the folder
3. `pip install -r requirements.txt`
4. Restart ComfyUI and Refresh the UI in your browser

## How it works
ChromaFlow uses the idea of partitioning to cluster the pixels from the motion guidance image into RGB space. It then walks through the clustered RGB space using the cluster centers as waypoints. On each step of the walk, it does a search for some number of "nearest neighbor pixels" being nearest in RGB color to the current point in the walk.
![image](https://github.com/lks-ai/chromaflow/assets/163685473/32c115dd-67cb-4c25-9558-6103a231474b)

### Motion Guidance Images
This is not necessarily a new concept, but it is somewhat untouched territory in ComfyUI. You can think of the motion guidance image as a gradient which defines the direction of motion at every point in the image. If high strenghts are used for ControlNet and IPAdapter, it will force the output animation to have the static structure of the motion guidance image while everything within those boundaries moves and interacts.
#### Example of using a picture as motion guidance
![image (152)](https://github.com/lks-ai/chromaflow/assets/163685473/e6d2b7d7-f35c-4852-ac28-abb3d732f459) ![chromaflow_00352](https://github.com/lks-ai/chromaflow/assets/163685473/b18cb700-5202-4e84-8d4b-33351ccde535)

## Features
- Generates ControlNet Depth Map, and a positive and negative Attention Mask for IPAdapter
- Beat Matching (Use `BPM Config` node to match a tempo at any frame rate)
- Lorentz Walk (Walks through the space using the famous Lorentz Attractor)
- Fully adjustable masks (Change your walk to get a different animation)

## Roadmap
- Fixing up the walk looping on certiain walks
- Allowing the cluster centroids to be adjustable
- A Clip Bin that lets you visually re-arrange your generated clips into a larger video
