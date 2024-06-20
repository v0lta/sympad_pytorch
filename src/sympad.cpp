#include <torch/extension.h>

#include <iostream>
using namespace torch::indexing;
using namespace std;

/**
 * Pads a 1-dimensional tensor symmetrically with zeros.
 * This is a helper function for _pad_symmetric.
 *
 * @param signal The input tensor to be padded.
 * @param padl The number of zeros to pad on the left side of the tensor.
 * @param padr The number of zeros to pad on the right side of the tensor.
 * @return The padded tensor.
 */
torch::Tensor _pad_symmetric_1d(torch::Tensor signal, pair<int, int> pad_tuple)
{   int padl = pad_tuple.first;
    int padr = pad_tuple.second;
    int dimlen = signal.size(0);
    if (padl > dimlen || padr > dimlen)
    {
        if (padl > dimlen)
        {
            signal = _pad_symmetric_1d(signal, make_pair(dimlen, 0));
            padl = padl - dimlen;
        }
        else
        {
            signal = _pad_symmetric_1d(signal, make_pair(0, dimlen));
            padr = padr - dimlen;
        }
        return _pad_symmetric_1d(signal, make_pair(padl, padr));
    }
    else
    {
        std::vector<torch::Tensor> cat_list = {signal};
        if (padl > 0)
        {
            cat_list.insert(cat_list.begin(), signal.index({Slice(None, padl)}).flip(0));
        }
        if (padr > 0)
        {
            cat_list.push_back(signal.index({Slice(-padr, None)}).flip(0));
        }
        return torch::cat(cat_list, 0);
    }
}


/**
 * Pads a given signal symmetrically along multiple dimensions.
 *
 * @param signal The input signal to be padded.
 * @param pad_lists A vector of pairs representing the padding amounts for each dimension.
 *                  Each pair contains the left and right padding amounts for a dimension.
 * @return The padded signal.
 * @throws std::invalid_argument if the input signal has fewer dimensions than the specified padding dimensions.
 */
torch::Tensor _pad_symmetric(torch::Tensor signal, vector<pair<int, int>> pad_lists)
{
    int pad_dims = pad_lists.size();
    if (signal.dim() < pad_dims)
    {
        throw std::invalid_argument("not enough dimensions to pad.");
    }

    int dims = signal.dim() - 1;
    reverse(pad_lists.begin(), pad_lists.end());
    for (int pos = 0; pos < pad_dims; pos++)
    {
        int current_axis = dims - pos;
        signal = signal.transpose(0, current_axis);
        signal = _pad_symmetric_1d(signal, pad_lists[pos]);
        signal = signal.transpose(current_axis, 0);
    }
    return signal;
}

PYBIND11_MODULE(sympad, m) {
  m.def("_pad_symmetric", &_pad_symmetric, "A function that pads a tensor symmetrically");
  m.def("_pad_symmetric_1d", &_pad_symmetric_1d, "A function that pads a tensor symmetrically in 1D.");
}
