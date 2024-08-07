#!/bin/bash

isSuccess=1

# Execute the mthreads-gmi command
mthreads_gmi_output=$(mthreads-gmi)

#Check if the mthreads-gmi output is empty
if [[ ! $mthreads_gmi_output =~ "Driver Version" ]]; then
    echo -e "\033[31mError! mthreads-gmi did not output anything! Please check the underlying drivers and container tools \033[0m"
    filesize=$(which mthreads-gmi | xargs ls -l | awk '{print $5}')
    isSuccess=0
    if [ "$filesize" -eq 0 ]; then
        echo -e "\033[31mIf the size of mthreads-gmi is 0, musa container tool is abnormal. \033[0m"
    fi
else
    echo -e "\033[32mmthreads-gmi command check successfully! \033[0m"
fi


#Check clinfo output
clinfo_output=$(clinfo | grep "Platform Vendor" | awk '{print $3 $4}')  
if [ "$clinfo_output" != "MooreThreads" ]; then  
    echo -e "\033[31mError! In the output of clinfo, the value of 'Platform Vendor' is not 'MooreThreads'! \033[0m"
    isSuccess=0
else 
    echo -e "\033[32mclinfo command check successfully! \033[0m"
fi

#Check simple musa program
mcc test_musa.mu -o test_musa -mtgpu -O2 -lmusart -L${MUSA_INSTALL_PATH}/lib
if [ $? -ne 0 ]; then  
    echo -e "\033[31mMCC failed to compile a simple demo musa program. The compilation command is:mcc test_musa.mu -o test_musa -mtgpu -O2 -lmusart -L${MUSA_INSTALL_PATH}/lib \033[0m"
    isSuccess=0
else
    ./test_musa
    if [ $? -eq 1 ]; then
        echo -e "\033[31mTest failed, MUSA code output does not meet expectations. MUSA source file:test_musa.mu. The compilation command is:mcc test_musa.mu -o test_musa -mtgpu -O2 -lmusart -L${MUSA_INSTALL_PATH}/lib \033[0m"
        isSuccess=0
    else    
        echo -e "\033[32msimple demo check successfully! \033[0m"
    fi
    rm ./test_musa
fi

echo " __  __  ___   ___  ____  _____   _____ _   _ ____  _____    _    ____  ____  "
echo "|  \/  |/ _ \ / _ \|  _ \| ____| |_   _| | | |  _ \| ____|  / \  |  _ \/ ___| "
echo "| |\/| | | | | | | | |_) |  _|     | | | |_| | |_) |  _|   / _ \ | | | \___ \ "
echo "| |  | | |_| | |_| |  _ <| |___    | | |  _  |  _ <| |___ / ___ \| |_| |___) |"
echo "|_|  |_|\___/ \___/|_| \_\_____|   |_| |_| |_|_| \_\_____/_/   \_\____/|____/ "
echo ""

if [ $isSuccess -eq 1 ]; then 
    echo " ____                              _ "
    echo "/ ___| _   _  ___ ___ ___  ___ ___| |"
    echo "\___ \| | | |/ __/ __/ _ \/ __/ __| |"
    echo " ___) | |_| | (_| (_|  __/\__ \__ \_|"
    echo "|____/ \__,_|\___\___\___||___/___(_)"
else 
    echo " _____     _ _          _ _ "
    echo "|  ___|_ _(_) | ___  __| | |"
    echo '| |_ / _` | | |/ _ \/ _` | |'
    echo "|  _| (_| | | |  __/ (_| |_|"
    echo "|_|  \__,_|_|_|\___|\__,_(_)"
fi
echo ""
exec /bin/bash
