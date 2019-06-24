package(default_visibility = ["//visibility:public"])

cc_library(
    name = "seastar",
    includes = [
        ".",
        "cached-fmt",
        "cached-c-ares",
        "cached-build/release/gen",
        "cached-build/release/c-ares",
    ],
    copts = [
        "-std=gnu++1y",
        "-DNO_EXCEPTION_HACK",
        "-DNO_EXCEPTION_INTERCEPT",
        "-DDEFAULT_ALLOCATOR",
        "-DHAVE_NUMA",
    ],
    linkopts = [
        "-L/usr/local/lib/",
        "-laio",
        "-lrt",
        "-lunwind",
        "-lnuma",
        "-ldl",
        "-lm",
    ],
    srcs = glob(
        ["**/*.cc"],
        exclude = [
            "cached-c-ares/c-ares/test/**",
            "cached-fmt/test/**",
            "cached-dpdk/**",
            "tests/**",
            "build/**",
            "apps/**",
            "rpc/**",
            "core/prometheus.cc",
            "net/proxy.cc",
            "net/virtio.cc",
            "net/dpdk.cc",
            "net/ip.cc",
            "net/ethernet.cc",
            "net/arp.cc",
            "net/native-stack.cc",
            "net/ip_checksum.cc",
            "net/udp.cc",
            "net/tcp.cc",
            "net/dhcp.cc",
            "net/tls.cc",
            "net/dns.cc",
            "core/dpdk_rte.cc",
            "cached-build/release/gen/proto/**",
        ],
    ),
    deps = [
        "@boost//:filesystem",
        "@boost//:thread",
        "@boost//:program_options",
        "@boost//:system",
    ],
)
