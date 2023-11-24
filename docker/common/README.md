- [Introduction](#Introduction)
- [Update Release Libraries](#update-release-libraries)
  - [Release MCCL](#release-mccl)
  - [Release MUDNN](#release-mudnn)
  - [Release MUSA Toolkits](#release-musa-toolkits)
  - [All Release Libraries](#all-release-libraries)
- [Update Daily Libraries](#update-daily-libraries)
  - [Daily MCCL](#daily-mccl)
  - [Daily MUDNN](#daily-mudnn)
  - [Daily MUSA Toolkits](#daily-musa-toolkits)
  - [All Daily Libraries](#all-daily-libraries)
- [Math Libraries](#math-libraries)
# Introduction
The scripts for updating the daily and release versions are located in their respective folders, namely, the "daily" folder and the "release" folder. Now the release version is [dev1.5.1](https://oss.mthreads.com:9001/buckets/release-rc/browse/Y3VkYV9jb21wYXRpYmxlL2RldjEuNS4xLw==).
<br>Note: [daily/update_daily_musart.sh](./daily/update_daily_musart.sh) is only used for torch_musa CI, please don't use it to update.

# Update Release Libraries
## **Release MCCL**  
  ```shell
  bash release/update_release_mccl.sh
  ```

## **Release MUDNN:**
  ```shell
  bash release/update_release_mudnn_temp.sh
  ```
 Please do not use docker/common/update_release_mudnn.sh for updates. It is specifically meant for the **torch_musa** CI (Continuous Integration) process.

## **Release MUSA Toolkits**
  ```shell
  bash release/update_release_musa_toolkits.sh
  ```
Note: Currently, the musa toolkits consist of the following components: musa_runtime, mcc, musify, muFFT, muBLAS, muPP, and muRAND.

## **All Release Libraries**
  ```shell
  bash release/update_release_all.sh
  ```
Note: This will update release MCCL, release MUDNN and release MUSA Toolkits at once.

# Update Daily Libraries
## **Daily MCCL**  
  ```shell
  bash daily/update_daily_mccl.sh
  ```

## **Daily MUDNN:**
  ```shell
  bash daily/update_daily_mudnn.sh
  ```

## **Daily MUSA Toolkits**
  ```shell
  bash daily/update_daily_musa_toolkits.sh
  ```
Note: Currently, the musa toolkits consist of the following components: musa_runtime, mcc, musify, muFFT, muBLAS, muPP, and muRAND.

## **All Daily Libraries**
  ```shell
  bash daily/update_daily_all.sh
  ```
Note: This will update daily MCCL, daily MUDNN and daily MUSA Toolkits at once.

# **Math Libraries**
```shell
bash install_math.sh [-w] [-c] [-r] [-s] [-a] [-t] [-b]

#view the help information
bash install_math.sh -h
```
Note: There is no separate release version for the math library and only the latest version is available.
