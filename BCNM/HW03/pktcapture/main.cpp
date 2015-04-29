#include "handle.h"


void my_callback(u_char *args,const struct pcap_pkthdr* pkthdr,const u_char*
        packet)
{
    u_int16_t type = handle_ethernet(args,pkthdr,packet);

    if(type == ETHERTYPE_IP)
    	/* handle IP packet */
        handle_IP(args,pkthdr,packet);
}


int main(int argc,char **argv)
{
  pcap_if_t *devlist, *device;
  char *dev, *filter;
  char errbuf[PCAP_ERRBUF_SIZE];
  pcap_t* descr;
  struct bpf_program fp;      /* hold compiled program     */
  bpf_u_int32 maskp;          /* subnet mask               */
  bpf_u_int32 netp;           /* ip                        */
  struct pcap_stat ps;
  u_char* args = NULL;
  int ch, npackets;

  dev = filter = NULL;
  npackets = -1;

	while((ch = getopt(argc, argv, "c:f:i:l")) != -1)	{
		switch(ch){
    case 'c':
			npackets = atoi(optarg);
			break;
		case 'f':
			filter = optarg;
			break;
		case 'i':
			dev = optarg;
			break;
		case 'l':
		    /* get the devices list*/
		    if(pcap_findalldevs(&devlist, errbuf) == -1){
		    	printf("%s\n", errbuf);
		    	exit(1);
		    }
		    /* list the device*/
		    for(device = devlist; device; device = device->next){
		    	printf("%s - %s\n", device->name, device->description);
		    }
		    pcap_freealldevs(devlist);
			break;
		default:
			fprintf(stdout,"Usage: %s numpackets\n",argv[0]);
			exit(1);
		}
	}

  if(dev == NULL)
  {
		dev = pcap_lookupdev(errbuf);
		if(dev == NULL)
		{ printf("%s\n",errbuf); exit(1); }
  }

  /* ask pcap for the network address and mask of the device */
  pcap_lookupnet(dev,&netp,&maskp,errbuf);

  /* open device for reading. NOTE: defaulting to
   * promiscuous mode*/
  descr = pcap_open_live(dev,BUFSIZ,1,-1,errbuf);
  if(descr == NULL)
  { printf("pcap_open_live(): %s\n",errbuf); exit(1); }

  if(filter != NULL)
  {
  	/* Lets try and compile the program.. non-optimized*/
    if(pcap_compile(descr, &fp, filter, 0, netp) == -1)
    { fprintf(stderr, "Error calling pcap_compile.\n"); exit(1); }

    /* set the compiled program as the filter */
    if(pcap_setfilter(descr, &fp) == -1)
    { fprintf(stderr, "Error setting filter\n"); exit(1); }
  }

  pcap_loop(descr, npackets, my_callback,args);

  if(pcap_stats(descr, &ps) == -1) {
  	printf("pcap_stats(): %s\n",errbuf);
  	exit(1);
  } else {
  	printf("\nrecv packet: %d drop packet:%d\n", ps.ps_recv, ps.ps_drop);
  }

  fprintf(stdout,"\nfinished\n");

	return 0;
}
