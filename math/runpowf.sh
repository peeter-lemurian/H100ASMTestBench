./pow_test --dump-inputs ./powtest.in
../torch/torchbinary.py --op pow --file ./powtest.in
./pow_test --torchinductor torchinductorpow.bin --torcheager torcheagerpow.bin --verbose --quiet --color | less -R
