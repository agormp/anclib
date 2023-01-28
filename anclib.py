import collections
import csv
import math
import re

import sequencelib
import phylotreelib as pt

import numpy as np
import pandas as pd
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

###################################################################################################
###################################################################################################

class AncError(Exception):
    pass

###################################################################################################
###################################################################################################

class Anc_recon():
    """Objects of this class contain a tree and a corresponding set of sequences.

    Some sequences correspond to leaves on the tree. These are observed sequences.
    Other sequences correspond to internal nodes on the tree. These are ancestral reconstructions.

    There is also information about credibility of sequence states: 100% for leaves, possibly
    less for internal nodes. If there is no information about credibility, then these are
    set to 100%.

    Objects may also contain information about other, non-sequence, traits. These also have
    associated credibilities.

    There are several constructors corresponding to different sources of information.
    There are also methods for adding extra information to already constructed objects.
    """

    def __init__(self):
        self.tree = None
        self.alignment = None
        self.seqprob = None
        self.trait = {}
        self.traitprob = {}
        self.sortedintnodes = None
        self.sortednodes = None

    ###############################################################################################

    @classmethod
    def from_baseml_mbasr(cls, baseml_rstfile,
                              mbasr_treefile, mbasr_intnodefile, mbasr_leafstatefile,
                              state0="state_0", state1="state_1"):
        """Combine information from BASEML rst file, and MBASR ancestral state reconstruction

        The following attributes are added:
            tree
            alignment of sequences, including ancestral reconstructions. Seqname = str(nodeid)
            probabilities for seq residues (dict of nodeid:numpy array)
            trait state (dict of nodeid:state)
            trait state probabillity (dict of nodeid:state probability)

        The nodeids used by MBASR are changed to match those used by BASEML
        """

        obj = cls()
        obj.__init__()
        ba_file = _Baseml_rstfile(baseml_rstfile)
        obj.tree = ba_file.get_tree()
        obj.alignment, obj.seqprob = ba_file.get_alignment()

        mb_file = _MBASR_file(mbasr_treefile, mbasr_intnodefile, mbasr_leafstatefile, state0, state1)
        mb_tree = mb_file.get_tree()
        mb_trait, mb_probs = mb_file.get_trait_dict()
        obj.sortedintnodes = sorted(list(obj.tree.intnodes))
        obj.sortednodes = sorted(list(obj.tree.leaves))
        obj.sortednodes.extend(obj.sortedintnodes)

        mbnode2banode, unmatch_baseml, unmatch_mbasr = mb_tree.match_nodes(obj.tree)
        if (unmatch_baseml != None) or (unmatch_mbasr != None):
            raise AncError("BASEML and MBASR trees are rooted differently. Cannot merge info")
        for mbnode,banode in mbnode2banode.items():
            obj.trait[banode] = mb_trait[mbnode]
            obj.traitprob[banode] = mb_probs[mbnode]
        return obj

    ###############################################################################################

    @classmethod
    def from_timetree(cls, ancseqfile, seqtreefile, statetreefile, stateprobfile):
        """Parse information from timetree ancestral reconstruction of sequences and state

        The following attributes are added:
            tree
            alignment of sequences, including ancestral reconstructions. Seqname = str(nodeid)
            trait state (dict of nodeid:state)

        While keeping track of the internal node IDs used in the two trees (probably identical...)
        """

        obj = cls()
        obj.__init__()
        tt = _TimeTreeResults(ancseqfile, seqtreefile, statetreefile, stateprobfile)
        obj.tree = tt.seqtree
        obj.alignment = tt.alignment
        obj.trait = tt.trait
        obj.traitprob = tt.traitprob
        obj.sortedintnodes = sorted(list(obj.tree.intnodes))
        obj.sortednodes = sorted(list(obj.tree.leaves))
        obj.sortednodes.extend(obj.sortedintnodes)
        return obj

    ###############################################################################################

    def nodeinfo(self, outfilename, varseq=False, poslist=None, zeroindex=True, probmin=None):
        """Writes results to 'outfilename':
        Output is one line per node in tree - with following informations:
            nodeid  traitstate  traitprob seqstates
        option varseq=True outputs only variable sites from sequences.
        option probmin (if not None): Only print info for nodes with traitprob > probmin
        Can also explicitly provide poslist of sites to be printed. Indexing in poslist
        can start at 0 (zeroindex=True) or 1 (zeroindex=False)"""

        with open(outfilename, "w") as outfile:
            outfile.write("# {}\t{}\t{}\t{}\n".format("nodeid", "trait", "traitprob", "seq"))
            pos = None
            if varseq and poslist:
                raise AncError("Specify either varseq or poslist option - not both")
            if varseq:
                pos = self.alignment.varcols()
            elif poslist:
                pos = poslist

            if pos and not zeroindex:
                pos = [p - 1 for p in pos]

            for nodeid in self.sortednodes:
                if (not probmin) or (self.traitprob[nodeid] > probmin):
                    seqname = str(nodeid)
                    trait = self.trait[nodeid]
                    traitprob = self.traitprob[nodeid]
                    if pos:
                        seq = self.alignment.getseq(seqname).subseqpos(pos).seq
                    else:
                        seq = self.alignment.getseq(seqname).seq
                    outfile.write(f"{nodeid}\t{trait}\t{traitprob:.3f}\t{seq}\n")

    ###############################################################################################

    def branchinfo(self, outfilename, varseq=False, poslist=None, zeroindex=True,
                    printif_traitdiff=False, printif_seqdiff=False, probmin=None):
        """Writes results to 'outfilename':
        Outputs one line per branch with following informations:
          nodeid_from  nodeid_to  trait_from  trait_to  traitprob_from  traitprob_to seq_from  seq_to
        Option varseq=True (default) output only variable sites from sequences.
        Option poslist: specify sites to be printed.
        Option zeroindex=True: Start indexing of poslist at 0 (otherwise start at 1)
        Option printif_traitdiff=True: only print branches where traits differ
        Option printif_seqdiff=True: only print branches where selected residues have changed
        option probmin (if not None): Only print info where both nodes have traitprob > probmin
        """

        with open(outfilename, "w") as outfile:
            outfile.write("# {}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                   "node_from", "node_to", "trait_from", "trait_to",
                   "traitprob_from", "traitprob_to", "seq_from", "seq_to"))
            pos = None
            if varseq and poslist:
                raise AncError("Specify either varseq or poslist option - not both")
            if varseq:
                pos = self.alignment.varcols()
            elif poslist:
                pos = poslist

            if pos and not zeroindex:
                pos = [p - 1 for p in pos]

            for nodefrom in self.sortedintnodes:
                for nodeto in self.tree.children(nodefrom):
                    if (not probmin) or (
                            (self.traitprob[nodefrom] > probmin) and
                            (self.traitprob[nodeto] > probmin)):
                        seqnamefrom = str(nodefrom)
                        seqnameto = str(nodeto)
                        traitfrom = self.trait[nodefrom]
                        traitto = self.trait[nodeto]
                        traitprobfrom = self.traitprob[nodefrom]
                        traitprobto = self.traitprob[nodeto]
                        if pos:
                            seqfrom = self.alignment.getseq(seqnamefrom).subseqpos(pos).seq
                            seqto = self.alignment.getseq(seqnameto).subseqpos(pos).seq
                        else:
                            seqfrom = self.alignment.getseq(seqnamefrom).seq
                            seqto = self.alignment.getseq(seqnameto).seq

                        # Could test for traitdiff before setting seq, but neater code this way...
                        printbranch = True
                        if printif_traitdiff and (traitfrom==traitto):
                            printbranch = False
                        if printif_seqdiff and (seqfrom==seqto):
                            printbranch = False
                        if printbranch:
                            outfile.write("{}\t{}\t{}\t{}\t{:.3f}\t{:.3f}\t{}\t{}\n".format(
                                   nodefrom, nodeto, traitfrom, traitto,
                                   traitprobfrom, traitprobto, seqfrom, seqto))

    ###############################################################################################

    def branchdiff(self, outfilename, zeroindex=True, printif_traitdiff=False, probmin=None):
        """Writes results to 'outfilename':
        Output is one line of output for each sequence change, on all branches of the tree:
            node_from   node_to   site  trait_from   trait_to   residue_from   residue_to
        Here 'site' is the index of the sequence residue.
        Option zeroindex=False causes numbering to start at 1 (otherwise at 0)
        Option printif_traitdiff=True: only print branches where traits differ
        Option probmin (if not None): Only print info where both nodes have traitprob > probmin
        """

        with open(outfilename, "w") as outfile:
            outfile.write("# {}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
               "node_from", "node_to", "seqpos", "trait_from", "trait_to",
               "traitprob_from", "traitprob_to", "residue_from", "residue_to"))
            for nodefrom in self.sortedintnodes:
                for nodeto in self.tree.children(nodefrom):
                    if (not probmin) or (
                            (self.traitprob[nodefrom] > probmin) and
                            (self.traitprob[nodeto] > probmin)):
                        seqnamefrom = str(nodefrom)
                        seqnameto = str(nodeto)
                        traitfrom = self.trait[nodefrom]
                        traitto = self.trait[nodeto]
                        traitprobfrom = self.traitprob[nodefrom]
                        traitprobto = self.traitprob[nodeto]
                        seqfrom = self.alignment.getseq(seqnamefrom)
                        seqto = self.alignment.getseq(seqnameto)
                        difflist = seqfrom.seqdiff(seqto, zeroindex)

                        # Could test for traitdiff before setting seq, but neater code this way...
                        if (not printif_traitdiff) or (traitfrom!=traitto):
                            for site,resfrom,resto in difflist:
                                outfile.write("{}\t{}\t{}\t{}\t{}\t{:.3f}\t{:.3f}\t{}\t{}\n".format(
                                   nodefrom, nodeto, site, traitfrom, traitto,
                                   traitprobfrom, traitprobto, resfrom, resto))

    ###############################################################################################

    def varpos(self, zeroindex=True):
        """Returns list of variable (unconserved) sites in alignment of sequences"""

        varpos = self.alignment.varcols()
        if not zeroindex:
            varpos = [pos+1 for pos in varpos]
        return varpos

###################################################################################################
###################################################################################################

class _Baseml_rstfile():
    """Class representing parser for BASEML rst file.

    Methods for extracing tree, alignment, and residue probabilities for ancestral
    (and contemporaneous) sequences.

    Tree is constructed based on branch info (87..88 87..89 etc.) and newick string.
    Internal nodes are numbered according to branch info. Leaves are named from Newick string.

    Ancestral sequences (corresponding to internal nodes on tree) are named based on nodeIDs.
    The method also returns
    """

    ###############################################################################################

    def __init__(self, filename=None):
        """Check file contains needed info about tree and sequences.
        Construct dictionaries mapping between nodeID, seqname, and index"""

        self.rstfile = open(filename, mode="rt", encoding="UTF-8")

        self._read_until("^Ancestral reconstruction by BASEML")
        self.rstfile.readline()
        self.newicknames = self.rstfile.readline()

        self.rstfile.readline()
        self.newicknumbers = self.rstfile.readline()

        # Find line with pattern like "   88..89  88..90"
        tmp,_ = self._read_until("^\s*[0-9]+\.\.[0-9]+\s*[0-9]+\.\.[0-9]+", nsave=1)
        self.branchlist = tmp[0].strip().split()

        tmp,_ = self._read_until("^Nodes [0-9]+ to [0-9]+ are ancestral", nsave=1)
        self.nseq = int(tmp[0].split()[3])

        # Find first line of seqinfo. Pattern like "   1     86   AAAAAAAAAAAAA"
        # Then find last line of seqinfo to get seqlen
        _,self.startofseqpos = self._read_until("^\s*1\s+[0-9]+\s+[A-Z]+")
        tmp,_ = self._read_until("^\n",nsave=2)
        self.seqlen = int(tmp[0].split()[0])

        # Note: nodenums start at 1, not 0 (this is BASEML node-numbering scheme)
        self.nodeid2nodenum, self.nodenum2nodeid = self._map_nodeid_nodenum()
        self.nodeids = set(self.nodeid2nodenum.keys())

    ###########################################################################################

    def close(self):
        self.rstfile.close()

    ###########################################################################################

    def _map_nodeid_nodenum(self):

        # Assumes the two newick strings in rst are in the same order
        leaflist = re.sub(",|;|\(|\)|: [0-9]\.[0-9]+", "", self.newicknames).split()
        numtmp = re.sub(",|;|\(|\)", "", self.newicknumbers).split()
        numberlist = [int(x) for x in numtmp]
        nodenum2nodeid = dict(zip(numberlist, leaflist))
        for nodenum in range(max(numberlist) + 1, self.nseq + 1):
            nodenum2nodeid[nodenum] = nodenum
        nodeid2nodenum = {val:key for key,val in nodenum2nodeid.items()}
        return (nodeid2nodenum, nodenum2nodeid)

    ###########################################################################################

    def get_tree(self):
        """Extracts tree from branch info and Newick string. Returns phylotreelib.Tree object"""

        parentlist, childlist = self._parsebranches()
        return pt.Tree.from_branchinfo(parentlist, childlist)

    ###########################################################################################

    def _parsebranches(self):
        parentlist = []
        childlist = []
        for branch in self.branchlist:
            parent,child = branch.split("..")
            parent = self.nodenum2nodeid[int(parent)]
            child = self.nodenum2nodeid[int(child)]
            parentlist.append(parent)
            childlist.append(child)
        return(parentlist, childlist)

    ###########################################################################################

    def get_alignment(self):
        """Extracts sequence information corresponding to extant and ancestral sequences.
        Returns sequencelib.Seq_alignment object and 2D numpy array with residue probs"""

        seqlists = [ [] for i in range(self.nseq)]
        seqprob = {nodeid:np.zeros(self.seqlen) for nodeid in self.nodeids}

        # Move filepointer to first line of sequence info
        self.rstfile.seek(self.startofseqpos)

        # Read seqinfo one line (=site) at a time, append residues and probs to relevant lists
        for line in self.rstfile:
            if line == "\n":
                break
            site,residues,probs = self._parse_siteline(line)
            for i,res in enumerate(residues):
                seqlists[i].append(res)
            for i,prob in enumerate(probs):
                nodenum = i + 1
                nodeid = self.nodenum2nodeid[nodenum]
                seqprob[nodeid][site-1] = prob

        # Determine seqtype from first seq (assume it is representative)
        seqtype = sequencelib.find_seqtype(seqlists[0])
        if seqtype not in ["DNA", "protein"]:
            raise AncError("Expecting either DNA or protein seqtype. Actual seqtype: {}".format(seqtype))

        # Create alignment from lists of residues in seqs
        alignment = sequencelib.Seq_alignment(seqtype=seqtype)
        for i in range(self.nseq):
            seq = "".join(seqlists[i])
            nodenum = i + 1
            nodeid = self.nodenum2nodeid[nodenum]
            name = str(nodeid)      # Leaves are already strings, so no effect on those
            if seqtype == "DNA":
                seqobject = sequencelib.DNA_sequence(name, seq)
            else:
                seqobject = sequencelib.Protein_sequence(name, seq)
            alignment.addseq(seqobject)
        return (alignment, seqprob)

    ###########################################################################################

    def _parse_siteline(self, line):
        words = line.split()
        site = int(words[0])
        leafres = words[2].replace(":","")
        residues = [residue for residue in leafres]
        probs = [1.00] * len(leafres)
        for word in words[3:]:
            word = word.replace(")","")
            residue,prob = word.split("(")
            prob = float(prob)
            residues.append(residue)
            probs.append(prob)
        return site,residues,probs

    ###########################################################################################

    def _read_until(self, regex, nsave=0):
        """Read filecontent up to and including regular expression pattern.
        Return last nsave lines if requested.
        Also returns position for start of last line (so can rewind one line using seek)"""

        lastnlines = collections.deque([None]*nsave)
        while True:
            prevlinepos = self.rstfile.tell()
            line = self.rstfile.readline()
            if line == "":
                msg = "BASEML rst file format not as expected"
                raise AncError(msg)
            if nsave > 0:
                lastnlines.append(line)
                lastnlines.popleft()
            if re.search(regex, line):
                return (lastnlines, prevlinepos)

###################################################################################################
###################################################################################################

class _MBASR_file():
    """Class representing parser for MBASR ancestral reconstruction file.

    Methods for extracing one tree and one alignment of ancestral (and contemporaneous) sequences.

    Tree is constructed based on branch info (87..88 87..89 etc.) and newick string.
    Internal nodes are numbered according to branch info. Leaves are named from Newick string.

    Ancestral sequences (corresponding to internal nodes on tree) are named based on nodeIDs.
    The method also returns
    """

    def __init__(self, treefile, intnodefile, leafstatefile, state0, state1):
        """Read files, store as dataframes, ready for further parsing.
        Tree is read as dataframe (so we have info about internal nodeIDs):
            parent, node, branch.length,	label
        Intnode reconstructions are read as dataframe:
            nodeid, pstate1, pstate2
        Leafnode states read as dataframe:
            leafnodeid, state
        """

        # Read treefile, store branch info as pandas dataframe
        # Note: I am cargo culting the rpy2 code here...
        # Format:
        ape = importr("ape")
        tidytree = importr("tidytree")
        tree = ape.read_tree(treefile)
        dftreeR = tidytree.as_tibble_phylo(tree)
        with localconverter(robjects.default_converter + pandas2ri.converter):
            self.dftreePy = robjects.conversion.rpy2py(dftreeR)
        self.nseq = self.dftreePy.shape[0]

        # Read internal node ancestral reconstruction state info
        self.intnodestate = pd.read_csv(intnodefile, delim_whitespace=True,
                                        header=None, names=["nodeID","pstate0","pstate1"])
        self.nintnode = self.intnodestate.shape[0]

        # Read leaf node state info
        self.leafnodestate = pd.read_csv(leafstatefile, delim_whitespace=True,
                                        header=None, names=["leafID","state"])
        self.nleaf = self.leafnodestate.shape[0]
        self.state0 = state0
        self.state1 = state1

    ###########################################################################################

    def get_tree(self):
        parentlist = []
        childlist = []

        for i in range(self.nseq):
            parent, node, label = self.dftreePy.iloc[i,[0,1,3]]
            # Tidy tree df has one row where root is both parent and node: Discard.
            if parent != node:
                parentlist.append(parent)
                # Tidytree parent is either leafname (type: string) or
                # NA_character_ (type: rpy2.rinterface_lib.sexp.NACharacterType)
                if type(label) == str:
                    childlist.append(label)
                else:
                    childlist.append(node)
        tree = pt.Tree.from_branchinfo(parentlist, childlist)
        return tree

    ###########################################################################################

    def get_trait_dict(self):
        statenamedict = {0:self.state0, 1:self.state1}
        nodeidlist = list(self.leafnodestate["leafID"])
        intnodeidlist = [int(x.replace("node","")) for x in self.intnodestate["nodeID"]]
        nodeidlist.extend(intnodeidlist)

        traitstatelist = [statenamedict[state] for state in self.leafnodestate["state"]]
        traitproblist = [1.0] * self.nleaf
        intnodestatelist = []
        intproblist = []
        for i in range(self.nintnode):
            p0, p1 = self.intnodestate.iloc[i,[1,2]]
            if p0 > 0.5:
                intnodestatelist.append(self.state0)
                intproblist.append(p0)
            else:
                intnodestatelist.append(self.state1)
                intproblist.append(p1)
        traitstatelist.extend(intnodestatelist)
        traitproblist.extend(intproblist)

        traitdict = dict(zip(nodeidlist, traitstatelist))
        traitprobdict = dict(zip(nodeidlist, traitproblist))

        return (traitdict, traitprobdict)

###################################################################################################
###################################################################################################

class _TimeTreeResults:

    def __init__(self, ancseqfile, seqtreefile, statetreefile, stateprobfile):
        self.seqtree, self.seqname2id = self._parsetreefile(seqtreefile)
        self.statetree, self.statename2id = self._parsetreefile(statetreefile)
        self.alignment = self._parsealignfile(ancseqfile, self.seqtree, self.seqname2id)
        self.check_input()
        statename2seqid = self._match_state_seq(self.statetree, self.seqtree, self.statename2id)
        self.trait, self.traitprob = self._parsetraits(statetreefile, stateprobfile,
                                                        statename2seqid, self.seqtree)

    ###########################################################################################

    def _parsetreefile(self, treefile):
        with pt.Nexustreefile(treefile) as tf:
            tree = tf.readtree()
            node2id = self._nodename_to_id(tree)
        return tree, node2id

    ###########################################################################################

    def _nodename_to_id(self, tree):
        name2id = {}
        for parent in tree.intnodes:
            for child in tree.children(parent):
                if child in tree.intnodes:
                    nodename = tree.getlabel(parent,child)
                else:
                    nodename = child
                name2id[nodename] = child
        # Add nodename for root (which was listed at end of treestring)
        # a bit hackish, since I know there is a nodename and a branchlen: NODE_0000000:0.00100
        rootname = re.sub(":[0-9]\.[0-9]+", "", tree.belowroot)
        name2id[rootname] = tree.root
        return name2id

    ###########################################################################################

    def _parsealignfile(self, alignfile, tree, name2id):
        """Reads alignment; returns alignment with nodes renamed to match nodes on seqtree"""
        sf = sequencelib.Seqfile(alignfile)
        alignment = sf.read_alignment()
        for origname in alignment.seqnamelist.copy():
            newname = str(name2id[origname])
            alignment.changeseqname(origname, newname)
        return alignment

    ###########################################################################################

    def _match_state_seq(self, statetree, seqtree, statename2id):
        stateid2seqid,_,_ = statetree.match_nodes(seqtree)
        statename2seqid = {}
        for statename, stateid in statename2id.items():
            seqid = stateid2seqid[stateid]
            statename2seqid[statename] = seqid
        return statename2seqid

    ###########################################################################################

    def _parsetraits(self, statetreefile, stateprobfile, statename2seqid, seqtree):
        trait = self._parse_annot_tree(statetreefile, statename2seqid) # This dict has no root entry
        traitprob, seqid2AB = self._parse_CSV(stateprobfile, statename2seqid)
        trait = self._match_AB_trait(seqid2AB, trait, seqtree)   # Add root trait. Check consistency
        return trait, traitprob

    ###########################################################################################

    def _parse_annot_tree(self, statetreefile, statename2seqid):
        with open(statetreefile, "r") as infile:
            while True:
                line = infile.readline()
                if line.startswith("Begin Trees"):
                    break
            treestring = infile.readline()
            while True:
                line = infile.readline()
                if line.startswith("End;"):
                    break
                else:
                    treestring += line
        treestring = treestring.replace("\n", "")
        nodenames = re.findall(r"[\(\),]([^\(\),:]+):", treestring)
        nodenames = nodenames[:-1]    # One too long: no state for root node for some reason
        states = re.findall(r'\[&[\w]+=\"(\w+)\"\]', treestring)
        traitdict = {}
        for name,state in zip(nodenames, states):
            seqid = statename2seqid[name]
            traitdict[seqid] = state
        return traitdict

    ###########################################################################################

    def _parse_CSV(self, stateprobfile, statename2seqid):
        traitprob = {}
        seqid2AB = {}
        with open(stateprobfile) as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                seqid = statename2seqid[row[0]]
                pA, pB = float(row[1]), float(row[2])
                if pA > pB:
                    traitprob[seqid] = pA
                    seqid2AB[seqid] = "A"
                else:
                    traitprob[seqid] = pB
                    seqid2AB[seqid] = "B"
        return traitprob, seqid2AB

    ###########################################################################################

    def _match_AB_trait(self, seqid2AB, trait, seqtree):
        # Python note: hardwired to two possible trait values. Think about generalizing
        random_leaf = seqtree.leaflist()[0]
        ABval = seqid2AB[random_leaf]
        traitval = trait[random_leaf]
        possible_traitvals = set(trait.values())
        other_trait = (possible_traitvals - {traitval}).pop()
        AB2trait = {}
        if ABval == "A":
            AB2trait["A"] = traitval
            AB2trait["B"] = other_trait
        else:
            AB2trait["A"] = other_trait
            AB2trait["B"] = traitval
        root_ABtrait = seqid2AB[seqtree.root]
        root_trait = AB2trait[root_ABtrait]
        trait[seqtree.root] = root_trait
        self._check_trait_consistency(AB2trait, seqid2AB, trait)
        return trait

    ###########################################################################################

    def _check_trait_consistency(self, AB2trait, seqid2AB, traitdict):
        for seqid,trait in traitdict.items():
            ABval = seqid2AB[seqid]
            expected_trait = AB2trait[ABval]
            if expected_trait != traitdict[seqid]:
                raise AncError(f"Inconsistency between confidence.csv and annotated tree for node {seqid}:\n"
                               f"Trait in CSV file:       {expected_trait}\n"
                               f"Trait in annotated tree: {trait[seqid]}")

    ###########################################################################################

    def check_input(self):
        if len(self.seqtree.nodes) != len(self.statetree.nodes):
            raise AncError("State-tree and seq-tree do not have same number of nodes. Can't merge")
        if self.seqtree.topology() != self.statetree.topology():
            raise AncError("State-tree and seq-tree do not have same topology. Can't merge")
        if len(self.alignment) != len(self.seqtree.nodes):
            raise AncError(f"Wrong number of sequences:"
                            " {len(self.alignment)} seqs in FASTA file, but"
                            " {len(self.seqtree)} nodes in seqtree")










