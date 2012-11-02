#! /bin/bash

#Check Package Requirements
PACKAGES="libx11-dev libxt-dev gfortran subversion"
dpkg -s $PACKAGES
if [[ $? -ne 0 ]]; then
  echo "Some Libraries are not installed. Lets fix that shall we?"
  apt-get install -y $PACKAGES
fi


PREFIX="/locale/ns"
if [ $VERSION == 35 ]; then
TCLVER=8.5.10
TKVER=8.5.10
OTCLVER=1.14
TCLCLVER=1.20
NSVER=2.35
NAMVER=1.15
XGRAPHVER=12.2
ZLIBVER=1.2.3
DEI80211MRVER=1.1.4

elif [ $VERSION == 33 ]; then

TCLVER=8.4.18
TKVER=8.4.18
OTCLVER=1.13
TCLCLVER=1.19
NSVER=2.33
NAMVER=1.13
XGRAPHVER=12.1
ZLIBVER=1.2.3
DEI80211MRVER=1.1.4

elif [ $VERSION == 34 ]; then

TCLVER=8.4.18
TKVER=8.4.18
OTCLVER=1.13
TCLCLVER=1.19
NSVER=2.34
NAMVER=1.13
XGRAPHVER=12.1
ZLIBVER=1.2.3
DEI80211MRVER=1.1.4

else
  echo "Version string makes no sense, bailing"
  exit

fi

WOSSVER=1.3.2
SUNSETVER=1.0

vercomp () {
    if [[ $1 == $2 ]]
    then
        return 0
    fi
    local IFS=.
    local i ver1=($1) ver2=($2)
    # fill empty fields in ver1 with zeros
    for ((i=${#ver1[@]}; i<${#ver2[@]}; i++))
    do
        ver1[i]=0
    done
    for ((i=0; i<${#ver1[@]}; i++))
    do
        if [[ -z ${ver2[i]} ]]
        then
            # fill empty fields in ver2 with zeros
            ver2[i]=0
        fi
        if ((10#${ver1[i]} > 10#${ver2[i]}))
        then
            return 1
        fi
        if ((10#${ver1[i]} < 10#${ver2[i]}))
        then
            return 2
        fi
    done
    return 0
}

# Generate Locale
mkdir -p ${PREFIX}

# Get NS Allinone and Install
if [ ! -x $PREFIX/ns-allinone-$NSVER/bin/ns ]; then 
  if [ ! -f $PREFIX/ns-ai1-$NSVER.tgz ]; then
    wget http://downloads.sourceforge.net/project/nsnam/allinone/ns-allinone-$NSVER/ns-allinone-$NSVER.tar.gz -O $PREFIX/ns-ai1-$NSVER.tgz 
  fi  
  rm -rf $PREFIX/ns-allinone-$NSVER
  tar -xzf $PREFIX/ns-ai1-$NSVER.tgz -C $PREFIX
  # Check if OTCL is less that 1.14 and if so try and apply patch from 'http://nsnam.isi.edu/nsnam/index.php/User_Information#Ubuntu_Installation_Guide'
  vercomp $OTCLVER 1.14
  if [[ $? == 2 ]]; then #lessthan
    echo "Patching OTcl version $OTCLVER"
    cd $PREFIX/ns-allinone-$NSVER/otcl-$OTCLVER/
    patch << HERE
--- configure
+++ configure.new
@@ -6301,7 +6301,7 @@
         ;;
     Linux*)
         SHLIB_CFLAGS="-fpic"
-        SHLIB_LD="ld -shared"
+        SHLIB_LD="gcc -shared"
         SHLIB_SUFFIX=".so"
         DL_LIBS="-ldl"
         SHLD_FLAGS=""
HERE
  fi
  # Check if ns is less that 2.35 and if so try and apply the ranvar.cc patch from 'http://wirelesscafe.wordpress.com/2009/05/28/error-with-ns2-installations-%E2%80%93-ns2-revisited/'
  vercomp $NSVER 2.35
  if [[ $? == 2 ]]; then #lessthan
    echo "Patching NS version $NSVER for ranvar.cc"
    cd $PREFIX/ns-allinone-$NSVER/ns-$NSVER/tools/
    patch --ignore-whitespace << HERE
--- ranvar.cc
+++ ranvar.cc.new
@@ -216,7 +216,7 @@
  // ACM Transactions on mathematical software, Vol. 26, No. 3, Sept. 2000
  if (alpha_ < 1) {
    double u = rng_->uniform(1.0);
-   return GammaRandomVariable::GammaRandomVariable(1.0 + alpha_, beta_).value() * pow (u, 1.0 / alpha_);
+   return GammaRandomVariable(1.0 + alpha_, beta_).value() * pow (u, 1.0 / alpha_);
  }
  
  double x, v, u;
HERE
    echo "Patching NS version $NSVER for mac-802_11Ext.h"
    cd $PREFIX/ns-allinone-$NSVER/ns-$NSVER/mac/
    patch --ignore-whitespace << HERE
--- mac-802_11Ext.h
+++ mac-802_11Ext.h.new
@@ -57,6 +57,7 @@
  
 #ifndef ns_mac_80211Ext_h
 #define ns_mac_80211Ext_h
+#include <cstddef>
 #include "marshall.h"
 #include "timer-handler.h"
 #define GET_ETHER_TYPE(x)    GET2BYTE((x))
HERE
    echo "Patching NS version $NSVER for nakagami.cc"
    cd $PREFIX/ns-allinone-$NSVER/ns-$NSVER/mobile/
patch --ignore-whitespace << HERE
--- nakagami.cc
+++ nakagami.cc.new
@@ -180,9 +180,9 @@
  double resultPower;
  
         if (int_m == m) {
- resultPower = ErlangRandomVariable::ErlangRandomVariable(Pr/m, int_m).value();
+ resultPower = ErlangRandomVariable(Pr/m, int_m).value();
  } else {
- resultPower = GammaRandomVariable::GammaRandomVariable(m, Pr/m).value();
+ resultPower = GammaRandomVariable(m, Pr/m).value();
  }
  return resultPower;
 }
HERE

  fi

  cd $PREFIX/ns-allinone-$NSVER/
    
  ./install || exit 1
fi
if [ ! -x $PREFIX/ns-allinone-$NSVER/bin/ns ]; then
  echo "Can't find ns executable after ns creation; installation probably failed in some weird and wonderful way..."
  exit 1
fi
# NS-Miracles 
if [ ! -d $PREFIX/nsmiracle-dev ]; then
  svn co --username nsmiracle-dev-guest --password nsmiracleguest https://telecom.dei.unipd.it:/tlcrepos/nsmiracle-dev/trunk $PREFIX/nsmiracle-dev
  cd $PREFIX/nsmiracle-dev/main/
  ./autogen.sh
  ./configure --with-ns-allinone=$PREFIX/ns-allinone-${NSVER} --prefix=$PREFIX --disable-static --with-dei80211mr=$PREFIX/ns-allinone-${NSVER}/dei80211mr-${DEI80211MRVER}
  make && make install || (make clean && exit 1)
fi

#Test must be run in the samples director due to local dependencies
cd $PREFIX/nsmiracle-dev/main/samples/
if [ -x $PREFIX/ns-allinone-$NSVER/bin/ns ]; then
  $PREFIX/ns-allinone-$NSVER/bin/ns dei80211mr_infrastruct_plus_wired_voip.tcl
else
  echo "Can't find ns executable; installating probably failed in some weird and wonderful way..."
  exit 1
fi


# WOSS
if [ ! -d ${PREFIX}/WOSS/at ]; then
  mkdir -p $PREFIX/WOSS/at
  #Bellhop Acoustic Toolbox
  wget -O - "http://oalib.hlsresearch.com/Modes/AcousticsToolbox/atLinux.tar.gz" | tar -xzf - -C $PREFIX/WOSS/
  cd $PREFIX/WOSS/at
  sed -i "s|/home/porter/at|/$PREFIX/WOSS/at|g" Makefile
  make clean
  make install || exit 1
fi

if [ ! -d $PREFIX/WOSS/woss ]; then
  #WOSS proper
  wget -O - "http://telecom.dei.unipd.it/ns/woss/files/WOSS-v$WOSSVER.tar.gz" | tar -xzf - -C $PREFIX/WOSS/
  cd $PREFIX/WOSS/
  ./autogen.sh
  ./configure --with-ns-allinone=$PREFIX/ns-allinone-${NSVER} --prefix=$PREFIX --with-pthread --with-nsmiracle=$PREFIX/lib
  make && make install || (make clean && exit 1)
fi

#SUNSET
if [ ! -d $PREFIX/SUNSET_v$SUNSETVER ]; then
  mkdir $PREFIX/SUNSET
  wget -O - "http://reti.dsi.uniroma1.it/UWSN_Group/framework/download/SUNSET_v$SUNSETVER.tar.gz" | tar -xzf - -C $PREFIX/
  cd $PREFIX/SUNSET_v$SUNSETVER/
  sed -i "s|NS_PATH=\"/home/example/\"|NS_PATH=\"$PREFIX/ns-allinone-$NSVER/\"|g" install_all.sh
  sed -i "s|MIRACLE_PATH=\"/home/example/\"|MIRACLE_PATH=\"$PREFIX/nsmiracle-dev/main/\"|g" install_all.sh
  ./install_all.sh || (make clean && exit 1)
  #Clean up the bashrc messing
  sed -i "/SUNSET_PATH/d" ~/.bashrc
  sed -i "s|pathMiracle \"insert_miracle_libraries_path_here\"|pathMiracle \"$PREFIX/lib\"|g" $PREFIX/SUNSET_v${SUNSETVER}/samples/*.tcl
  sed -i "s|pathWOSS \"insert_woss_libraries_path_here\"|pathWOSS \"$PREFIX/WOSS/\"|g" $PREFIX/SUNSET_v${SUNSETVER}/samples/*.tcl
  sed -i "s|pathSUNSET \"insert_sunset_libraries_path_here\"|pathSUNSET \"$PREFIX/lib\"|g" $PREFIX/SUNSET_v${SUNSETVER}/samples/*.tcl
  #Test must be run in the samples director due to local dependencies
  cd $PREFIX/SUNSET_v$SUNSETVER/samples/
  $PREFIX/ns-allinone-$NSVER/bin/ns runSimulationBellhop.tcl
fi

rm /etc/ld.so.conf.d/sunset.conf /etc/profile.d/sunset.conf

echo >> /etc/ld.so.conf.d/sunset.conf << END
$PREFIX/ns-allinone-$NSVER/otcl-$OTCLVER
$PREFIX/ns-allinone-$NSVER/lib
$PREFIX/lib
END
echo >> /etc/profile.d/sunset.conf << END
export PATH="$PREFIX/ns-allinone-$NSVER/bin:$PREFIX/ns-allinone-$NSVER/tcl$TCLVER/unix:$PREFIX/ns-allinone-2.35/tk$TKVER/unix:$PATH"
export TCL_LIBRARY="$PREFIX/ns-allinone-$NSVER/tcl$TCLVER/library,$TCL_LIBRARY"
export SUNSET_PATH="$PREFIX/SUNSET_v$SUNSERVER"
END



