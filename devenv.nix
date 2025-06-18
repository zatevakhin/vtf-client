{
  pkgs,
  lib,
  ...
}: {
  languages.python = {
    enable = true;
    version = "3.10";
    venv.enable = true;
    venv.requirements = builtins.readFile ./requirements.txt;
  };

  packages = with pkgs; [
    zlib
    glib
    libGL
    libva
    stdenv.cc.cc
    xorg.libxcb
    xorg.libX11
    xorg.libXext
    xorg.libXrender
    xorg.libSM
    xorg.libICE
    qt5.qtbase
    websocat
  ];

  env = {
    LD_LIBRARY_PATH = lib.makeLibraryPath [
      pkgs.stdenv.cc.cc
    ];

    QT_PLUGIN_PATH = "${pkgs.qt5.qtbase}/${pkgs.qt5.qtbase.qtPluginPrefix}";
    QT_QPA_PLATFORM = "xcb";
  };
}
