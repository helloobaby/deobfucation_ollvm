#Backspace
#Backspace
from binaryninja import *
import pprint
from operator import *

def safe_asm(bv, asm_str):
    return bv.arch.assemble(asm_str)

def llil_at(bv, addr):
    funcs = bv.get_functions_containing(addr)
    if not funcs:
        return None

    return funcs[0].get_low_level_il_at(addr)

def is_call(bv, addr):
    llil = llil_at(bv, addr)
    if llil is None:
        return False

    return llil.operation == LowLevelILOperation.LLIL_CALL

class CFGLink(object):
    def __init__(self, block, true_block, false_block=None, def_il=None):
        """ Create a link from a block to its real successors

        Args:
            block (BasicBlock): block to start from
            true_block (BasicBlock): The target block of an unconditional jump,
                or the true branch of a conditional jump
            false_block (BasicBlock): The false branch of a conditional jump
            def_il (MediumLevelILInstruction): The instruction that was used
                to discover this link. This will be a definition of the state
                variable
        """
        self.il = def_il  # The definition il we used to find this link
        self.block = block

        # Resolve the true/false blocks
        self.true_block = true_block.outgoing_edges[0].target
        self.false_block = false_block
        if self.false_block is not None:
            self.false_block = self.false_block.outgoing_edges[0].target

    @property
    def is_uncond(self):
        return self.false_block is None

    @property
    def is_cond(self):
        return not self.is_uncond

    def gen_asm(self, bv, base_addr):
        """ Generates a patch to repair this link

        For an unconditional jump, this will generate
            jmp next_block

        For a conditional jump, this will generate
            jcc true_block
            jmp false_block
        where cc is the condition used in the original CMOVcc in the flattening logic

        Args:
            bv (BinaryView)
            base_addr (int): The address where these instructions will be placed.
                This is necessary to calculate relative addresses

        Returns:
            str: The assembled patch opcodes
        """
        # It's assumed that base_addr is the start of free space
        # at the end of a newly recovered block
        def rel(addr):
            return hex(addr - base_addr).rstrip('L')

        # Unconditional jmp
        if self.is_uncond:
            next_addr = self.true_block.start
            print ('[+] Patching from {:x} to {:x}'.format(base_addr, next_addr))
            return safe_asm(bv, 'jmp {}'.format(rel(next_addr)))

        # Branch based on original cmovcc
        else:
            assert self.il is not None
            true_addr = self.true_block.start
            false_addr = self.false_block.start
            print ('[+] Patching from {:x} to T: {:x} F: {:x}'.format(base_addr,
                                                                     true_addr,
                                                                     false_addr))

            # Find the cmovcc by looking at the def il's incoming edges
            # Both parent blocks are part of the same cmov
            il_bb = next(bb for bb in self.il.function if bb.start <= self.il.instr_index < bb.end)
            cmov_addr = il_bb.incoming_edges[0].source[-1].address
            cmov = bv.get_disassembly(cmov_addr).split(' ')[0]

            # It was actually painful to write this
            jmp_instr = cmov.replace('cmov', 'j')

            # Generate the branch instructions
            asm = safe_asm(bv, '{} {}'.format(jmp_instr, rel(true_addr)))
            base_addr += len(asm)
            asm += safe_asm(bv, 'jmp {}'.format(rel(false_addr)))

            return asm

    def __repr__(self):
        if self.is_uncond:
            return '<U Link: {} => {}>'.format(self.block,
                                               self.true_block)
        else:
            return '<C Link: {} => T: {}, F: {}>'.format(self.block,
                                                         self.true_block,
                                                         self.false_block)


# 获得一个地址所在的函数
def get_func_containing(bv, addr):
    """ Finds the function, if any, containing the given address """
    funcs = bv.get_functions_containing(addr)
    return funcs[0] if funcs else None

def compute_original_blocks(bv, mlil, state_var):
    """ Gathers all MLIL instructions that (re)define the state variable
    Args:
        bv (BinaryView)
        mlil (MediumLevelILFunction): The MLIL for the function to be deflattened
        state_var (Variable): The state variable in the MLIL

    Returns:
        tuple: All MediumLevelILInstructions in mlil that update state_var
    """
    original = mlil.get_var_definitions(state_var)
    return itemgetter(*original)(mlil)


def compute_backbone_map(bv, mlil, state_var):
    """ Recover the map of state values to backbone blocks

    This will generate a map of
    {
        state1 => BasicBlock1,
        state2 => BasicBlock2,
        ...
    }

    Where BasicBlock1 is the block in the backbone that will dispatch to
    an original block if the state is currently equal to state1

    Args:
        bv (BinaryView)
        mlil (MediumLevelILFunction): The MLIL for the function to be deflattened
        state_var (Variable): The state variable in the MLIL

    Returns:
        dict: map of {state value => backbone block}
    """
    backbone = {}

    # The state variable itself isn't always the one referenced in the
    # backbone blocks, they may instead use another pointer to it.
    # Find the variable that all subdispatchers use in comparisons
    var = state_var
    uses = mlil.get_var_uses(var)
    # 原理是SSA形式的use-def链
    # 大于2的话很可能在子分发块了
    while len(uses) <= 2: 
        var = mlil[uses[-1]].dest
        uses = mlil.get_var_uses(var)

    logger.log_info(f'每个子分发块的循环调度变量{var}')

    uses += mlil.get_var_definitions(var) # 
    logger.log_info(f'子分发块的分发变量{uses}')

    # Gather the blocks where this variable is used
    blks = (b for il in uses for b in mlil.basic_blocks if b.start <= il.instr_index < b.end)
    # In each of these blocks, find the value of the state

    # 只能处理以下类似的模式
    # bool cond:1_1 = temp0_1 == 0xa2ca54f7
    # if (temp0_1 == 0x91aa9163) then 17 @ 0x400d0b else 20 @ 0x4009dd
    for bb in blks:
        # Find the comparison
        #logger.log_info(f'bb[-1] -> {bb[-1]}')
        
        if '==' in str(bb[-1].operands[0]):  # 直接跟常数比较的模式
            state = bb[-1].operands[0].right.value.value
            backbone[state] = bv.get_basic_blocks_at(bb[0].address)[0]
        else:
            cond_var = bb[-1].condition.src
            cmp_il = mlil[mlil.get_var_definitions(cond_var)[0]]

            # Pull out the state value
            state = cmp_il.src.right.constant
            backbone[state] = bv.get_basic_blocks_at(bb[0].address)[0]

    return backbone

def resolve_cfg_link(bv, mlil, il, backbone):
    bb = bv.get_basic_blocks_at(il.address)[0]
    
    # StateVar = 0xb01971f0
    # 这种常量赋值
    if il.src.operation == MediumLevelILOperation.MLIL_CONST or il.src.operation == MediumLevelILOperation.MLIL_CONST_PTR:
        return CFGLink(bb, backbone[il.src.constant], def_il=il)
    else:
        # Go into SSA to figure out which state is the false branch
        # Get the phi for the state variable at this point
        phi = get_ssa_def(mlil, il.ssa_form.src.src)
        assert phi.operation == MediumLevelILOperation.MLIL_VAR_PHI

        # The cmov (select) will only ever replace the default value (false)
        # with another if the condition passes (true)
        # So all we need to do is take the earliest version of the SSA var
        # as the false state
        f_def, t_def = sorted(phi.src, key=lambda var: var.version)

        # There will always be one possible value here
        false_state = get_ssa_def(mlil, f_def).src.possible_values.value
        true_state  = get_ssa_def(mlil, t_def).src.possible_values.value

        logger.log_info(f'false_state : {false_state:#x} true_state:{true_state:#x}')

        return CFGLink(bb, backbone[true_state], backbone[false_state], il)

def clean_block(bv, mlil, link):
    """ Return the data for a block with all unnecessary instructions removed

    Args:
        bv (BinaryView)
        mlil (MediumLevelILFunction): The MLIL for the function to be deflattened
        link (CFGLink): a link with the resolved successors for a block

    Returns:
        str: A copy of the block link is based on with all dead instructions removed
    """

    # Helper for resolving new addresses for relative calls
    def _fix_call(bv, addr, newaddr):
        tgt = llil_at(bv, addr).dest.constant
        reladdr = hex(tgt - newaddr).rstrip('L')
        return safe_asm(bv, 'call {}'.format(reladdr))

    # The terminator gets replaced anyway
    block = link.block
    old_len = block.length
    nop_addrs = {block.disassembly_text[-1].address}

    # Gather all addresses related to the state variable
    if link.il is not None:
        gather_defs(link.il.ssa_form, nop_addrs)

    # Rebuild the block, skipping the bad instrs
    addr = block.start
    data = b''
    while addr < block.end:
        # How much data to read
        ilen = bv.get_instruction_length(addr)

        # Only process this instruction if we haven't blacklisted it
        if addr not in nop_addrs:
            # Calls need to be handled separately to fix relative addressing
            if is_call(bv, addr):
                data += _fix_call(bv, addr, block.start + len(data))
            else:
                data += bv.read(addr, ilen)

        # Next instruction
        addr += ilen
    return data, block.start + len(data), old_len
    
def gather_full_backbone(backbone_map):
    """ Collect all blocks that are part of the backbone

    Args:
        backbone_map (dict): map of {state value => backbone block}

    Returns:
        set: All BasicBlocks involved in any form in the backbone
    """
    # Get the immediately known blocks from the map
    backbone_blocks = list(backbone_map.values())
    backbone_blocks += [bb.outgoing_edges[1].target for bb in backbone_blocks]

    # Some of these blocks might be part of a chain of unconditional jumps back to the top of the backbone
    # Find the rest of the blocks in the chain and add them to be removed
    for bb in backbone_blocks:
        blk = bb
        while len(blk.outgoing_edges) == 1:
            if blk not in backbone_blocks:
                backbone_blocks.append(blk)
            blk = blk.outgoing_edges[0].target
    return set(backbone_blocks)

def gather_defs(il, defs):
    """ Walks up a def chain starting at the given il (mlil-ssa)
    until constants are found, gathering all addresses along the way
    """
    defs.add(il.address)
    op = il.operation

    if op == MediumLevelILOperation.MLIL_CONST:
        return

    if op in [MediumLevelILOperation.MLIL_VAR_SSA_FIELD,
              MediumLevelILOperation.MLIL_VAR_SSA]:
        gather_defs(get_ssa_def(il.function, il.src), defs)

    if op == MediumLevelILOperation.MLIL_VAR_PHI:
        for var in il.src:
            gather_defs(get_ssa_def(il.function, var), defs)

    if hasattr(il, 'src') and isinstance(il.src, MediumLevelILInstruction):
        gather_defs(il.src, defs)

def get_ssa_def(mlil, var):
    """ Gets the IL that defines var in the SSA form of mlil """
    return mlil.ssa_form.get_ssa_var_definition(var)

# 测试入口点
logger = Logger(session_id=0, logger_name=__name__)
logger.log_info('[ollvm]Plugin Run')

# 第一个基本块
# int64_t var_10 = rbx
# void* rsi = &var_88
# int64_t rcx = [fsbase + 0x28].q
# int64_t var_18 = rcx
# int32_t var_90 = 0
# rax = read(fd: 0, buf: rsi, nbytes: 0x64)
# void* rdi = &var_88
# ssize_t var_a0 = rax
# rax_1, r8, r9 = sub_3170(rdi)
# int32_t var_8c = rax_1
# int32_t var_94 = 0xa2ca54f7 -> current_address 赋值StateVar
# goto 12 @ 0x52e5

func = get_func_containing(bv, current_address)
mlil = func.medium_level_il
state_var = func.get_low_level_il_at(current_address).medium_level_il.dest
logger.log_info('[ollvm]ollvm 的调度状态变量 -> ' + state_var.name)

backbone = compute_backbone_map(bv,mlil,state_var)
logger.log_info('提取到的StateVar与真实块的map')
logger.log_info(backbone)

original = compute_original_blocks(bv, mlil, state_var)
logger.log_info('提取到的StateVar赋值的块')
logger.log_info(f'{original}')

CFG = [resolve_cfg_link(bv, mlil, il, backbone) for il in original]
print(CFG)

for link in CFG:
        if link==None:
            continue
        # Clean out instructions we don't need to make space
        blockdata, cave_addr, orig_len = clean_block(bv, mlil, link)

        # Add the new instructions and patch, nop the rest of the block
        blockdata += link.gen_asm(bv, cave_addr)
        blockdata = blockdata.ljust(orig_len, safe_asm(bv, 'nop'))
        bv.write(link.block.start, blockdata)

    # Do some final cleanup
nop = safe_asm(bv, 'nop')
for bb in gather_full_backbone(backbone):
    print ('[+] NOPing block: {}'.format(bb))
    bv.write(bb.start, nop * bb.length)