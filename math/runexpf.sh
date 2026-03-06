./expf_test --dump-inputs ./exptest.in
../torch/torchunary.py --op exp --file ./exptest.in
./expf_test --torchinductor torchinductorexp.bin --torcheager torcheagerexp.bin --verbose --quiet --color | less -R
