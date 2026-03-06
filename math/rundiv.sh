./div_test --dump-inputs ./divtest.in
../torch/torchbinary.py --op div --file ./divtest.in
./div_test --torchinductor torchinductordiv.bin --torcheager torcheagerdiv.bin --verbose --quiet --color | less -R
