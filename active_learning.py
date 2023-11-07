from scipy.spatial.distance import cdist

class ActiveLearning():
    def __init__(self):
        pass
    def similarity(self, z_images, z_text):
        '''
        Return a N x M matrix of cosine similarities between N image and M text embeddings
        z_images: a list of N embeddings of images
        z_text: a list of M embeddings of text
        return: the cosine similarity for the N x M pairs of image-text embeddings
        '''
        # Sklearn cosine distance computation is faster than cosine similarity
        cosine_dist = cdist(z_images, z_text, metric = 'cosine') 
        return 1 - cosine_dist # cosine similarity = 1 - cosine_distance
    def get_s_score(self, z_images, z_text):
        '''
        Get the S-Score (Paper: https://ieeexplore.ieee.org/abstract/document/8629116) of a set of images and text embeddings.
        z_images: a list of N embeddings of images
        z_text: a list of M embeddings of text
        return: the N (image,text) similarities which indicate the most similar an image is to any of the texts
        '''
        # Compute the N x M cosine similarity between N image and M text embeddings
        sim = similarity(z_images, z_text)
        # Get the N highest similarities of the M x N text-image similarities, that indicates how similar (at most) the image is to any of the texts
        return  np.max(sim, axis = 1)
    def request_labels(self, z_images, z_text, bottom_N, method = 's-score'):
        '''
        Based on the S-Score, return the indices of the images which are least similar to any of the text embeddings.
        z_images: a list of N embeddings of images
        z_text: a list of M embeddings of text
        return: the indices of the images which are least similar to any of the text embeddings
        '''
        if method == 's-score':
            s_score = get_s_score(z_images, z_text)
            # Return the indices of the bottom N (= most dissimilar to the text embeddings) image embeddings
            return s_score.argsort()[:bottom_N]
        