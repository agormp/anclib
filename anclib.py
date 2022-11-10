###################################################################################################
###################################################################################################
###################################################################################################


class Baseml_rstfile():
    """Class representing BASEML rst file. 
    One single tree can be constructed based on branch info (87..88 87..89 etc.) and newick string.
    Internal nodes are numbered according to branch info. Leaves are named from Newick string.
    """
    
    # Note: different from other treefile classes. 
    # Iteration not implemented - should I for consistency?

    ###############################################################################################

    def __init__(self, filename=None, filecontent=None):
        """Check file contains needed info about tree. Find and store that info"""

        num_args = (filename is not None) + (filecontent is not None)
        if num_args != 1:
            raise TreeError("Baseml_rstfile __init__ requires either filename or filecontent (not both)")
        elif filecontent:
            self.treefile = StringIO(filecontent)
        else:
            self.treefile = open(filename, mode="rt", encoding="UTF-8")

        # Assume file format is similar to example below
        # Read first 17 lines, check format, and extract required information
        filestart = [self.treefile.readline() for i in range(17)]
        if (not "Supplemental results for BASEML" in filestart[0]) or (not "tree with node labels for Rod" in filestart[16]):
            msg = "File does not appear to be a BASEML rst file with ancestral reconstructions"
            raise TreeError(msg)
        else:
            self.branchlist = filestart[14].split()
            self.newicknames = filestart[10]
            self.newicknumbers = filestart[12]

        # Supplemental results for BASEML
        #
        # seqf:  NS_test_aln_trimmed_renamed.fa
        # treef: renamed_NS_test.nwk
        #
        #
        # TREE #  1
        #
        # Ancestral reconstruction by BASEML.
        #
        # ((seq_84: 0.094193, ((seq_25: 0.049725, (seq_67: 0.017366, ((seq_55: 0.011849, ((seq_81: ....
        #
        # ((84, ((25, (67, ((55, ((81, 36), ((83, 72), (((((((8, (20, 19)), 23), 26), 31), (((70,  ...
        #
        #   87..88   88..84   88..89   89..90   90..25   90..91   91..67   91..92   92..93         ...
        #
        # tree with node labels for Rod Page's TreeView
        # ((84_seq_84, ((25_seq_25, (67_seq_67, ((55_seq_55, ((81_seq_81, 36_seq_36) 95 ,          ...
        
    ###########################################################################################
    # Could also extract branch lengths, but slightly difficult and not sure it is ever useful?

    def read_tree(self):
        parentlist, childlist = self.parsebranches()
        leafdict = self.parsenewick()
        newchildlist = []
        for child in childlist:
            if child in leafdict:
                newchildlist.append(leafdict[child])
            else:
                newchildlist.append(child)
        return Tree.from_branchinfo(parentlist, newchildlist)
        
    ###########################################################################################

    def parsebranches(self):
        parentlist = []
        childlist = []
        for branch in self.branchlist:
            parent,child = branch.split("..")
            parentlist.append(int(parent))
            childlist.append(int(child))
        return(parentlist, childlist)
        
    ###########################################################################################

    # Assume the two newick strings are in the same order
    def parsenewick(self):
        namelist = re.sub(",|;|\(|\)|: [0-9]\.[0-9]+", "", self.newicknames).split()
        numtmp = re.sub(",|;|\(|\)", "", self.newicknumbers).split()
        numberlist = [int(x) for x in numtmp]
        leafdict = dict(zip(numberlist, namelist))
        return(leafdict)

    ###########################################################################################

    # Function for parsing rst to get changes for each branch? Does this not fit in phylotreelib?

###################################################################################################
###################################################################################################
###################################################################################################

