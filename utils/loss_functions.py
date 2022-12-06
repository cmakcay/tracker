import torch


def landmark_loss(actual_lmk, predicted_lmk):
    """
    Calculate landmark loss

    Parameters
    ----------
    actual_lmk
        Ground truth landmarks with size [N, 68, 2].
    predicted_lmk
        Predicted landmarks with size [N, 68, 2].
    """
    # calculate the l1 distance between ground truth landmarks and predicted landmarks
    lmk_norm = torch.abs(actual_lmk - predicted_lmk).sum(dim=2)
    avg_lmk_loss = torch.mean(lmk_norm)

    # different weights for different types of landmarks, nose tip and mouth corners:3; other mouth and nose:1.5;
    # the rest is 1
    # indices: nose: [27,35]; nose tip:30, 31, 35; mouth: [48, 67]; mouth corners: outer lip{48, 54} inner lip{60, 64}
    weights = torch.ones(68).cuda() #remove .cuda() when not using cuda
    weights[27:36] = 1.5
    weights[30] = 3.
    weights[31] = 3.
    weights[35] = 3.
    weights[48:68] = 2.5
    weights[48] = 3.
    weights[54] = 3.
    weighted_lmk_norm = torch.multiply(lmk_norm, weights)
    avg_lmk_loss_weighted = torch.mean(weighted_lmk_norm)
    return avg_lmk_loss, avg_lmk_loss_weighted


def eye_closure_loss(actual_lmk, predicted_lmk):
    """
    Calculate eye closure loss

    Parameters
    ----------
    actual_lmk
        Ground truth landmarks with size [N, 68, 2].
    predicted_lmk
        Predicted landmarks with size [N, 68, 2].
    """
    # calculate the difference between upper eyelid and lower eyelid
    # left eye upper eyelid 37, 38, lower eyelid 40, 41; right eye upper 43, 44, lower 46, 47
    # note the order on lower eyelid (clockwise)

    # Euler distance between eyelids
    eye_closure_actual = actual_lmk[:, list([37, 38, 43, 44]), :] - actual_lmk[:, list([41, 40, 47, 46]), :]
    eye_dist_actual = torch.sqrt(torch.sum(eye_closure_actual**2, 2))

    eye_closure_predict = predicted_lmk[:, list([37, 38, 43, 44]), :] - predicted_lmk[:, list([41, 40, 47, 46]), :]
    eye_dist_predict = torch.sqrt(torch.sum(eye_closure_predict**2, 2))

    eye_clos_loss = torch.mean(torch.abs(eye_dist_actual-eye_dist_predict))
    return eye_clos_loss
