import instaloader
import os
import re

def download_instagram_media(url, target_dir="temp"):
    # Ensure target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    L = instaloader.Instaloader(dirname_pattern=os.path.join(target_dir, "{shortcode}"), 
                                download_pictures=True, 
                                download_videos=True, 
                                download_video_thumbnails=False,
                                download_geotags=False,
                                download_comments=False,
                                save_metadata=False)
    
    # Extract shortcode from URL
    # https://www.instagram.com/p/C6R_Xy_S7Y_/ -> C6R_Xy_S7Y_
    # https://www.instagram.com/reel/C6R_Xy_S7Y_/ -> C6R_Xy_S7Y_
    match = re.search(r'/(?:p|reel)/([^/?#&]+)', url)
    if not match:
        return None, "Invalid Instagram URL"
    
    shortcode = match.group(1)
    
    try:
        post = instaloader.Post.from_shortcode(L.context, shortcode)
        L.download_post(post, target=shortcode)
        
        # Find the downloaded file
        download_path = os.path.join(target_dir, shortcode)
        if not os.path.exists(download_path):
            return None, "Download failed: directory not found"
            
        files = os.listdir(download_path)
        
        # Prefer video if available, then image
        video_files = [f for f in files if f.endswith('.mp4')]
        image_files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        
        if video_files:
            return os.path.abspath(os.path.join(download_path, video_files[0])), "video"
        elif image_files:
            return os.path.abspath(os.path.join(download_path, image_files[0])), "image"
        else:
            return None, "No media found in the post"
            
    except Exception as e:
        return None, str(e)
