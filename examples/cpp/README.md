This is a c++ deployment demo with torch_musa and you can also read Pytorch tutorial about [TorchScript](https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html)

### Build Source
```
bash build.sh
```

### Run Model
```
python model_gen.py 
./build/example-app "resnet50.pt"
# -2.0971 -0.9347 -0.3347 -0.6047 -0.9165
# [ PrivateUse1FloatType{1,5} ]
```
