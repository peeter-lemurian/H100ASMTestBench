./math/div_test --dump-inputs ./divtest.in
../bin/torchbinary.py --op div --file ./divtest.in
./math/div_test --torchinductor torchinductordiv.bin --torcheager torcheagerdiv.bin --verbose --quiet --color | less -R
