#!/bin/bash

# Execute the mthreads-gmi command
mthreads_gmi_output=$(mthreads-gmi)

#Check if the mthreads-gmi output is empty
if [[ ! $mthreads_gmi_output =~ "Driver Version" ]]; then
    echo -e "\033[31m Error! mthreads-gmi did not output anything! Please check the underlying drivers and container tools \033[0m"
    exec /bin/bash
fi
echo -e "\033[32mmthreads-gmi command check successfully! \033[0m"

#Check clinfo output
clinfo_output=$(clinfo | grep "Platform Vendor" | awk '{print $3 $4}')  
if [ "$clinfo_output" != "MooreThreads" ]; then  
    echo "Error! In the output of clinfo, the value of 'Platform Vendor' is not 'MooreThreads'!"
    exec /bin/bash
fi
echo -e "\033[32mclinfo command check successfully! \033[0m"

#Check simple musa program
mcc test_musa.mu -o test_musa -mtgpu -O2 -lmusart -L${MUSA_INSTALL_PATH}/lib
if [ $? -ne 0 ]; then  
    echo "MCC failed to compile a simple demo musa program. The compilation command is:mcc test_musa.mu -o test_musa -mtgpu -O2 -lmusart -L${MUSA_INSTALL_PATH}/lib"
    exec /bin/bash
fi

test_musa_output=$(./test_musa)
if [ "$test_musa_output" == "error!" ]; then
    echo "Test failed, MUSA code output does not meet expectations. MUSA source file:test_musa.mu. The compilation command is:mcc test_musa.mu -o test_musa -mtgpu -O2 -lmusart -L${MUSA_INSTALL_PATH}/lib" >&2
    exec /bin/bash
fi
rm ./test_musa

echo " __  __  ___   ___  ____  _____   _____ _   _ ____  _____    _    ____  ____  "
echo "|  \/  |/ _ \ / _ \|  _ \| ____| |_   _| | | |  _ \| ____|  / \  |  _ \/ ___| "
echo "| |\/| | | | | | | | |_) |  _|     | | | |_| | |_) |  _|   / _ \ | | | \___ \ "
echo "| |  | | |_| | |_| |  _ <| |___    | | |  _  |  _ <| |___ / ___ \| |_| |___) |"
echo "|_|  |_|\___/ \___/|_| \_\_____|   |_| |_| |_|_| \_\_____/_/   \_\____/|____/ "

echo " _____ _____ ____ _____   ____  _   _  ____ ____ _____ ____ ____  "
echo "|_   _| ____/ ___|_   _| / ___|| | | |/ ___/ ___| ____/ ___/ ___| "
echo "  | | |  _| \___ \ | |   \___ \| | | | |  | |   |  _| \___ \___ \ "
echo "  | | | |___ ___) || |    ___) | |_| | |__| |___| |___ ___) |__) |"
echo -e "  |_| |_____|____/ |_|   |____/ \___/ \____\____|_____|____/____/ \n"
