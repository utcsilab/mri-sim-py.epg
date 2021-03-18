import torch
from simulator import FSE_signal_TR


def FIM_T2_and_M0(angles_rad, TE, TR, M0, T1, T2):
    # _T2 = T2.clone().requires_grad_()
    # _M0 = M0.clone().requires_grad_()
    def myfun(m0, t2):
        # return m0[:,None,None] * FSE_signal_driveq(angles_rad, TE, TR, T1, t2)
        return m0[:, None, None] * FSE_signal_TR(angles_rad, TE, TR, T1, t2)

    batch_size = T2.shape[0]
    dZ = torch.autograd.functional.jacobian(myfun, (M0, T2), create_graph=True)
    dSdM0 = dZ[0]
    dSdT2 = dZ[1]
    F1 = torch.sum(dSdT2 ** 2, dim=1)
    F2 = torch.sum(dSdT2 * dSdM0, dim=1)
    F3 = torch.sum(dSdM0 ** 2, dim=1)
    FIM = torch.zeros([batch_size, 2, 2], device=T2.device)
    for i in range(batch_size):
        FIM[i, 0, 0] = F1[i, 0, i]
        FIM[i, 0, 1] = F2[i, 0, i]
        FIM[i, 1, 0] = F2[i, 0, i]
        FIM[i, 1, 1] = F3[i, 0, i]
    return FIM


def FIM_T2(angles_rad, TE, TR, M0, T1, T2):
    _T2 = T2.clone().requires_grad_()

    def myfun(t2):
        return FSE_signal_TR(angles_rad, TE, TR, T1, t2)

    dSdT2 = torch.autograd.functional.jacobian(myfun, _T2, create_graph=True)
    FIM = torch.sum(dSdT2 ** 2)
    return FIM


def CRLB_T2(angles_rad, TE, TR, M0, T1, T2):
    # return 1 / FIM_T2(angles_rad, TE, TR, M0, T1, T2)
    return torch.inverse(FIM_T2_and_M0(angles_rad, TE, TR, M0, T1, T2))[:, 0, 0]