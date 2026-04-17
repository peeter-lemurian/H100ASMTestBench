./math/expf_test --dump-inputs ./exptest.in
../bin/torchunary.py --op exp --file ./exptest.in
./math/expf_test --torchinductor torchinductorexp.bin --torcheager torcheagerexp.bin --verbose --quiet --color | less -R
