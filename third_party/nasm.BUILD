# Description:
#   NASM is a portable assembler in the Intel/Microsoft tradition.

licenses(["notice"])  # BSD 2-clause

exports_files(["LICENSE"])

cc_binary(
    name = "nasm",
    srcs = [
        "asm/assemble.c",
        "asm/assemble.h",
        "asm/directbl.c",
        "asm/directiv.c",
        "asm/directiv.h",
        "asm/error.c",
        "asm/eval.c",
        "asm/eval.h",
        "asm/exprdump.c",
        "asm/exprlib.c",
        "asm/float.c",
        "asm/float.h",
        "asm/labels.c",
        "asm/listing.c",
        "asm/listing.h",
        "asm/nasm.c",
        "asm/parser.c",
        "asm/parser.h",
        "asm/pptok.c",
        "asm/pptok.h",
        "asm/pragma.c",
        "asm/preproc.c",
        "asm/preproc.h",
        "asm/preproc-nop.c",
        "asm/quote.c",
        "asm/quote.h",
        "asm/rdstrnum.c",
        "asm/segalloc.c",
        "asm/stdscan.c",
        "asm/stdscan.h",
        "asm/strfunc.c",
        "asm/tokens.h",
        "asm/tokhash.c",
        "common/common.c",
        "config/unknown.h",
        "disasm/disasm.c",
        "disasm/disasm.h",
        "disasm/sync.c",
        "disasm/sync.h",
        "include/compiler.h",
        "include/disp8.h",
        "include/error.h",
        "include/hashtbl.h",
        "include/iflag.h",
        "include/insns.h",
        "include/labels.h",
        "include/md5.h",
        "include/nasm.h",
        "include/nasmint.h",
        "include/nasmlib.h",
        "include/opflags.h",
        "include/perfhash.h",
        "include/raa.h",
        "include/rbtree.h",
        "include/rdoff.h",
        "include/saa.h",
        "include/strlist.h",
        "include/tables.h",
        "include/ver.h",
        "macros/macros.c",
        "nasmlib/badenum.c",
        "nasmlib/bsi.c",
        "nasmlib/crc64.c",
        "nasmlib/file.c",
        "nasmlib/file.h",
        "nasmlib/filename.c",
        "nasmlib/hashtbl.c",
        "nasmlib/ilog2.c",
        "nasmlib/malloc.c",
        "nasmlib/md5c.c",
        "nasmlib/mmap.c",
        "nasmlib/path.c",
        "nasmlib/perfhash.c",
        "nasmlib/raa.c",
        "nasmlib/rbtree.c",
        "nasmlib/readnum.c",
        "nasmlib/realpath.c",
        "nasmlib/saa.c",
        "nasmlib/srcfile.c",
        "nasmlib/string.c",
        "nasmlib/strlist.c",
        "nasmlib/ver.c",
        "nasmlib/zerobuf.c",
        "output/codeview.c",
        "output/dwarf.h",
        "output/elf.h",
        "output/legacy.c",
        "output/nulldbg.c",
        "output/nullout.c",
        "output/outaout.c",
        "output/outas86.c",
        "output/outbin.c",
        "output/outcoff.c",
        "output/outdbg.c",
        "output/outelf.c",
        "output/outelf.h",
        "output/outform.c",
        "output/outform.h",
        "output/outieee.c",
        "output/outlib.c",
        "output/outlib.h",
        "output/outmacho.c",
        "output/outobj.c",
        "output/outrdf2.c",
        "output/pecoff.h",
        "output/stabs.h",
        "stdlib/snprintf.c",
        "stdlib/strlcpy.c",
        "stdlib/strnlen.c",
        "stdlib/vsnprintf.c",
        "version.h",
        "x86/disp8.c",
        "x86/iflag.c",
        "x86/iflaggen.h",
        "x86/insnsa.c",
        "x86/insnsb.c",
        "x86/insnsd.c",
        "x86/insnsi.h",
        "x86/insnsn.c",
        "x86/regdis.c",
        "x86/regdis.h",
        "x86/regflags.c",
        "x86/regs.c",
        "x86/regs.h",
        "x86/regvals.c",
    ] + select({
        ":windows": ["config/msvc.h"],
        "//conditions:default": [],
    }),
    includes = [
        "asm",
        "include",
        "output",
        "x86",
    ],
    copts = select({
        ":windows": [],
        "//conditions:default": [
            "-w",
            "-std=c99",
        ],
    }),
    defines = select({
        ":windows": [],
        "//conditions:default": [
            "HAVE_SNPRINTF",
            "HAVE_SYS_TYPES_H",
        ],
    }),
    visibility = ["@jpeg//:__pkg__"],
)

config_setting(
    name = "windows",
    values = {
        "cpu": "x64_windows",
    },
)
